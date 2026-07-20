"""
calcul_ET.py
============
Calcul de l'Évapotranspiration (ET) via le modèle TTME
(Two-source Trapezoid Model for Evapotranspiration)
Basé sur Long & Singh (2012).

Le modèle décompose la température radiométrique (LST) en composantes
sol (Ts) et canopée (Tc) via un espace trapézoïdal LST-fc, puis
calcule les flux d'énergie séparément pour chaque composante.

Phases :
  1. Préparation des variables d'entrée (LST, NDVI→fc, Ta, Rn, u)
  2. Calcul des limites théoriques (Boundary Conditions)
  3. Décomposition de la température (Ts, Tc par pixel)
  4. Paramétrisation séparée des flux (H, LE sol + canopée)
  5. Synthèse : ET totale pixel par pixel
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer

sys.stdout.reconfigure(encoding='utf-8')

from config import LOGGER, SITES_PILOTES, OUTPUT_DIR, TIME_MARGIN_MINUTES, SITE_TTME_PARAMS

# =============================================================================
# CONSTANTES PHYSIQUES
# =============================================================================
RHO_CP    = 1200.0     # ρ * c_p : capacité thermique volumique de l'air (J·m⁻³·K⁻¹)
SIGMA     = 5.67e-8    # Constante de Stefan-Boltzmann (W·m⁻²·K⁻⁴)
K_VK      = 0.41       # Constante de von Kármán
LAMBDA_V  = 2.45e6     # Chaleur latente de vaporisation (J·kg⁻¹) ~2.45 MJ/kg
Z_M       = 2.0        # Hauteur de mesure du vent (m)
Z_H       = 2.0        # Hauteur de mesure de la température (m)

# Paramètres supprimés car gérés dynamiquement via SITE_TTME_PARAMS

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def extract_datetime_from_filename(filename):
    """Extrait la date et l'heure d'un nom de fichier TIF."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})(?:_(\d{2})h(\d{2}))?", filename)
    if match:
        date_str = match.group(1)
        hour_str = match.group(2) if match.group(2) else "10"
        min_str = match.group(3) if match.group(3) else "30"
        return pd.to_datetime(f"{date_str} {hour_str}:{min_str}:00")
    return None


def compute_fc(ndvi, params):
    """Calcule la fraction de couverture végétale à partir du NDVI."""
    ndvi_sol = params.get("NDVI_SOL", 0.15)
    ndvi_veg = params.get("NDVI_VEG", 0.90)
    fc = ((ndvi - ndvi_sol) / (ndvi_veg - ndvi_sol)) ** 2
    return np.clip(fc, 0.0, 1.0)


def compute_aerodynamic_resistance(u, z0m, z0h, z_m=Z_M):
    """
    Calcule la résistance aérodynamique (s/m) en conditions de stabilité neutre.
    r_ah = ln(z_m/z0m) * ln(z_h/z0h) / (k² * u)
    """
    u = np.maximum(u, 0.5)  # Vitesse du vent minimale pour éviter division par 0
    r_ah = (np.log(z_m / z0m) * np.log(Z_H / z0h)) / (K_VK**2 * u)
    return r_ah


# =============================================================================
# CHARGEMENT DES DONNÉES MÉTÉO
# =============================================================================

def load_meteo_icos(site, target_dt, margin_min=60):
    """Charge les données météo depuis les CSV ICOS enrichis."""
    csv_path = os.path.join("Outputs_ICOS", f"donnees_icos_{site}.csv")
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    
    # Trouver la mesure la plus proche dans la marge de tolérance
    diff = abs(df['TIMESTAMP'] - target_dt)
    mask = diff <= pd.Timedelta(minutes=margin_min)
    if not df[mask].empty:
        idx_best = diff[mask].idxmin()
        row = df.loc[idx_best]
        return {
            'Ta':  row.get('TA_Consolide', np.nan),
            'u':   row.get('WS_Consolide', np.nan),
            'Rn':  row.get('Rn_Consolide', np.nan),
            'R_s_down': row.get('SW_IN_Consolide', np.nan),
            'R_l_down': row.get('LW_IN_Consolide', np.nan),
            'G':   row.get('G_Consolide', np.nan),
            'RH':  row.get('RH_Consolide', np.nan),
            'LST_ground': row.get('LST_Calculee', np.nan),
            'source': 'ICOS'
        }
    return None


def load_meteo_noaa(site, target_dt, margin_min=60):
    """Charge les données météo depuis les CSV NOAA enrichis."""
    csv_path = os.path.join("Outputs_NOAA", f"donnees_noaa_{site}.csv")
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    first_col = df.columns[0]
    df.rename(columns={first_col: 'TIMESTAMP'}, inplace=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']).dt.tz_localize(None)
    
    diff = abs(df['TIMESTAMP'] - target_dt)
    mask = diff <= pd.Timedelta(minutes=margin_min)
    if not df[mask].empty:
        idx_best = diff[mask].idxmin()
        row = df.loc[idx_best]
        return {
            'Ta':  row.get('TA_Consolide', np.nan),
            'u':   row.get('WS_Consolide', np.nan),
            'Rn':  row.get('Rn_Consolide', np.nan),
            'G':   np.nan,  # NOAA n'a pas G
            'RH':  row.get('RH_Consolide', np.nan),
            'LST_ground': row.get('LST_Calculee', np.nan),
            'source': 'NOAA'
        }
    return None


def load_meteo_gol(site, target_dt, margin_min=60):
    """Charge les données météo depuis les CSV GOL."""
    site_map = {"Italy": "Italie", "Greece": "Grece"}
    site_suffix = site_map.get(site, site)
    gol_file = os.path.join("donnees_Gol", f"temperatures_{site_suffix}.csv")
    
    if not os.path.exists(gol_file):
        return None
    
    df = pd.read_csv(gol_file)
    df['created_date'] = pd.to_datetime(df['created_date'], utc=True).dt.tz_localize(None)
    
    diff = abs(df['created_date'] - target_dt)
    mask = diff <= pd.Timedelta(minutes=margin_min)
    if not df[mask].empty:
        idx_best = diff[mask].idxmin()
        row = df.loc[idx_best]
        
        # Calcul du Rn depuis les 4 composantes radiatives GOL
        sw_in = row.get('field_highwavedn_avg', np.nan)
        sw_out = row.get('field_highwaveup_avg', np.nan)
        lw_in = row.get('field_lowwavedn_avg', np.nan)
        lw_out = row.get('field_lowwaveup_avg', np.nan)
        rn = np.nan
        if pd.notna(sw_in) and pd.notna(sw_out) and pd.notna(lw_in) and pd.notna(lw_out):
            rn = (sw_in - sw_out) + (lw_in - lw_out)
        
        return {
            'Ta':  row.get('field_tair_c_avg', np.nan),
            'u':   row.get('field_ws_ms', np.nan),
            'Rn':  rn,
            'G':   np.nan,
            'RH':  row.get('field_rh_avg', np.nan),
            'LST_ground': np.nan,
            'source': 'GOL'
        }
    return None


def load_meteo_era5(site, target_dt, margin_min=60):
    """Charge les données météo depuis les CSV ERA5."""
    csv_path = os.path.join("Outputs_ERA5", f"donnees_era5_{site}.csv")
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    first_col = df.columns[0]
    if first_col != 'TIMESTAMP':
        df.rename(columns={first_col: 'TIMESTAMP'}, inplace=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']).dt.tz_localize(None)
    
    diff = abs(df['TIMESTAMP'] - target_dt)
    mask = diff <= pd.Timedelta(minutes=margin_min)
    if not df[mask].empty:
        idx_best = diff[mask].idxmin()
        row = df.loc[idx_best]
        return {
            'Ta':  row.get('TA_Consolide', np.nan),
            'u':   row.get('WS_Consolide', np.nan),
            'Rn':  row.get('Rn_Consolide', np.nan),
            'R_s_down': row.get('SW_IN_Consolide', np.nan),
            'R_l_down': row.get('LW_IN_Consolide', np.nan),
            'G':   np.nan,
            'RH':  row.get('RH_Consolide', np.nan),
            'LST_ground': row.get('LST_Calculee', np.nan),
            'source': 'ERA5'
        }
    return None


def load_meteo_era5_ds(site, date_str, target_dt, margin_min=60):
    """Charge les données météo ERA5 et remplace la Ta par la Ta downscalée (1050m)."""
    meteo = load_meteo_era5(site, target_dt, margin_min)
    if meteo is None:
        return None
    
    # Exécuter / Charger le downscaling
    from Traitement.downscaling_Ta import process_site_date
    import rioxarray
    
    tif_path = process_site_date(site, date_str)
    if tif_path and os.path.exists(tif_path):
        try:
            coords = SITES_PILOTES[site]
            ds = rioxarray.open_rasterio(tif_path).squeeze()
            transformer = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
            x_utm, y_utm = transformer.transform(coords['lon'], coords['lat'])
            val = ds.sel(x=x_utm, y=y_utm, method='nearest').values
            if np.isfinite(val):
                meteo['Ta'] = float(val)
                meteo['source'] = 'ERA5_DS'
        except Exception as e:
            LOGGER.error(f"   ❌ Erreur extraction Ta downscalée pour {site} : {e}")
            
    return meteo
    return meteo

def load_meteo_era5_biais(site, target_dt, margin_min=60):
    meteo = load_meteo_era5(site, target_dt, margin_min)
    if meteo is None:
        return None
        
    BIASES = {
        "Gebesee": -0.81,
        "Selhausen": -1.54,
        "Lonzee": 0.68,
        "Voulundgaard": -1.61,
        "Klingenberg": 1.17,
        "Estrees-Mons": 1.27,
        "Borgo Cioffi": -0.88,
        "Grignon": 1.00,
        "Lamasquere": 6.91,
    }
    
    if site in BIASES and pd.notna(meteo['Ta']):
        meteo['Ta'] = meteo['Ta'] - BIASES[site]
        meteo['source'] = 'ERA5_BIAIS'
        LOGGER.info(f"      🌡️ Biais corrigé pour {site} (Correction: {-BIASES[site]:.2f}°C) -> Nouvelle Ta: {meteo['Ta']:.2f}°C")
        
    return meteo

def load_meteo(site, target_dt, date_str, source='icos'):
    """
    Charge les données météo depuis la source choisie.
    """
    if source == 'era5':
        meteo = load_meteo_era5(site, target_dt)
    elif source == 'era5_biais':
        meteo = load_meteo_era5_biais(site, target_dt)
    elif source == 'era5_ds':
        meteo = load_meteo_era5_ds(site, date_str, target_dt)
    else:
        # Mode ICOS : priorité ICOS > NOAA > GOL
        meteo = load_meteo_icos(site, target_dt)
        if meteo is None or pd.isna(meteo['Ta']):
            meteo = load_meteo_noaa(site, target_dt)
        if meteo is None or pd.isna(meteo['Ta']):
            meteo = load_meteo_gol(site, target_dt)
        
    # Si les flux radiatifs ne sont pas fournis par la station, on les récupère depuis ERA5
    if meteo is not None and pd.notna(meteo['Ta']):
        if pd.isna(meteo.get('R_s_down')) or pd.isna(meteo.get('R_l_down')):
            meteo_era5 = load_meteo_era5(site, target_dt)
            if meteo_era5 is not None:
                if pd.isna(meteo.get('R_s_down')):
                    meteo['R_s_down'] = meteo_era5.get('R_s_down', np.nan)
                if pd.isna(meteo.get('R_l_down')):
                    meteo['R_l_down'] = meteo_era5.get('R_l_down', np.nan)
        return meteo
    
    return None


# =============================================================================
# CŒUR DU MODÈLE TTME
# =============================================================================

def calculer_bilan_radiatif(bandes_landsat, fc, lst_celsius, R_s_down, R_l_down):
    """
    Calcule le Rayonnement Net (Rn) à 30m.
    
    Paramètres :
    - bandes_landsat : dictionnaire contenant les réflectances de surface Landsat 8/9
                       ex: {'B2': bleu, 'B4': rouge, 'B5': nir, 'B6': swir1, 'B7': swir2}
    - fc : Fraction de couverture végétale (numpy array, de 0 à 1)
    - lst_celsius : Température de surface issue du DMS en degrés Celsius
    - R_s_down : Rayonnement solaire descendant d'ERA5 (SSRD, converti en W/m²)
    - R_l_down : Rayonnement infrarouge thermique descendant d'ERA5 (STRD, converti en W/m²)
    
    Retourne :
    - Rn : Rayonnement net en W/m² (numpy array)
    """
    
    # 1. Calcul de l'Albédo (Méthode de Liang, 2001 pour Landsat)
    # Les bandes doivent être en réflectance de surface (0 à 1)
    alpha = (0.356 * bandes_landsat['B2'] + 
             0.130 * bandes_landsat['B4'] + 
             0.373 * bandes_landsat['B5'] + 
             0.085 * bandes_landsat['B6'] + 
             0.072 * bandes_landsat['B7'] - 0.0018)
             
    # Borner l'albédo pour éviter les valeurs aberrantes sur l'eau ou l'ombre
    alpha = np.clip(alpha, 0.01, 0.99)
    
    # 2. Calcul de l'Émissivité de surface (epsilon)
    # Basé sur une pondération classique entre un sol nu typique (0.971) et une canopée (0.989)
    epsilon = 0.971 * (1 - fc) + 0.989 * fc
    
    # 3. Conversion de la température en Kelvin
    lst_kelvin = lst_celsius + 273.15
    
    # 4. Calcul du Rayonnement Net (Rn)
    # Équation : Rn = (1 - alpha)*Rs_down + epsilon*Rl_down - epsilon*sigma*T^4
    Rn = (1 - alpha) * R_s_down + epsilon * R_l_down - epsilon * SIGMA * (lst_kelvin ** 4)
    
    return Rn

def ttme_compute_et(lst_array, ndvi_array, Ta, u, Rn, R_s_down, R_l_down, params, G_measured=None, transform=None, crs=None, z_m=Z_M):
    """
    Implémente le modèle TTME (Two-source Trapezoid Model for Evapotranspiration).
    
    Paramètres :
        lst_array   : np.ndarray 2D - LST en °C (ex: DMS sharpened à 30m)
        ndvi_array  : np.ndarray 2D - NDVI
        Ta          : float - Température de l'air (°C)
        u           : float - Vitesse du vent (m/s)
        Rn          : np.ndarray 2D - Rayonnement net (W/m²) spatialisé
        R_s_down    : float - Rayonnement solaire descendant d'ERA5
        R_l_down    : float - Rayonnement thermique descendant d'ERA5
        params      : dict - Dictionnaire des paramètres TTME du site
        G_measured  : float or None - Flux de chaleur sol mesuré (W/m²)
        transform   : rasterio.Affine - transform géospatial
        crs         : str - système de coordonnées
    
    Retourne :
        dict avec les résultats (arrays 2D) :
            'fc', 'Ts', 'Tc', 'Hs', 'Hc', 'LEs', 'LEc', 'LE', 'ET_mm_h',
            'EF', 'T_s_max', 'T_c_max', 'transform', 'crs'
    """
    
    # Extraire les paramètres
    z0m_s = params.get("Z0M_SOIL", 0.005)
    z0h_s = params.get("Z0H_SOIL", 0.0005)
    z0m_c = params.get("Z0M_VEG", 0.10)
    z0h_c = params.get("Z0H_VEG", 0.01)
    cg_s  = params.get("C_G_SOIL", 0.30)
    cg_c  = params.get("C_G_VEG", 0.05)
    
    # =========================================================================
    # PHASE 1 : Préparation des Variables d'Entrée
    # =========================================================================
    
    # Masque de validité
    valid = np.isfinite(lst_array) & np.isfinite(ndvi_array) & (lst_array > -50) & (lst_array < 80)
    
    # Fraction de couverture végétale
    fc = np.where(valid, compute_fc(ndvi_array, params), np.nan)
    
    # Vitesse du vent minimale
    u = max(u, 0.5)
    
    # =========================================================================
    # PHASE 2 : Calcul des Limites Théoriques (Boundary Conditions)
    # =========================================================================
    
    # Résistances aérodynamiques (neutralité supposée)
    r_ah_s = compute_aerodynamic_resistance(u, z0m_s, z0h_s, z_m=z_m)  # Sol nu
    r_ah_c = compute_aerodynamic_resistance(u, z0m_c, z0h_c, z_m=z_m)    # Canopée
    
    # Correction de stabilité simplifiée pour le bord chaud (convection libre)
    # On bride les résistances car la turbulence thermique les détruit à haute température.
    r_ah_s = min(r_ah_s, 110.0)
    r_ah_c = min(r_ah_c, 30.0)
    
    # Les bilans radiatifs des patchs "purs" utilisent le Rn global du pixel
    # On évite ainsi la double pondération lors de la recombinaison finale
    
    # Flux de chaleur dans le sol pour les patchs purs
    if G_measured is not None and pd.notna(G_measured):
        Gs = np.full_like(Rn, G_measured)
        Gc = cg_c * Rn
        G = np.where(valid, G_measured * (1.0 - fc) + Gc * fc, np.nan)
    else:
        Gs = cg_s * Rn
        Gc = cg_c * Rn
        G = np.where(valid, Gs * (1.0 - fc) + Gc * fc, np.nan)
    
    # --- Limite Froide (Lower Boundary) ---
    # T_s,min = T_c,min = Ta (toute l'énergie part en évaporation)
    T_s_min = Ta
    T_c_min = Ta
    
    # --- Limite Chaude (Upper Boundary) ---
    # Sol nu totalement sec (LE = 0) : H_s = Rn_theo_s - G_s
    # On force l'albédo à 0.25 et l'émissivité à 0.971 pour le sol nu sec, et on itère
    T_s_max_iter = Ta + 10.0
    for _ in range(3):
        Rn_s_theo = (1 - 0.25) * R_s_down + 0.971 * R_l_down - 0.971 * SIGMA * ((T_s_max_iter + 273.15)**4)
        T_s_max_iter = Ta + r_ah_s * (Rn_s_theo * (1.0 - cg_s)) / RHO_CP
    T_s_max = T_s_max_iter
    
    # Canopée totalement sèche (LE = 0) : H_c = Rn_theo_c - G_c
    # On force l'albédo à 0.20 et l'émissivité à 0.989 pour la canopée
    T_c_max_iter = Ta + 5.0
    for _ in range(3):
        Rn_c_theo = (1 - 0.20) * R_s_down + 0.989 * R_l_down - 0.989 * SIGMA * ((T_c_max_iter + 273.15)**4)
        T_c_max_iter = Ta + r_ah_c * (Rn_c_theo * (1.0 - cg_c)) / RHO_CP
    T_c_max = T_c_max_iter
    
    # Pente de la ligne chaude (warm edge) dans l'espace (fc, T)
    beta_w = T_c_max - T_s_max
    
    LOGGER.info(f"      📐 Limites théoriques : T_s,max = {T_s_max:.1f}°C | T_c,max = {T_c_max:.1f}°C | β_w = {beta_w:.2f}")
    LOGGER.info(f"      🌡️ Ta = {Ta:.1f}°C | u = {u:.1f} m/s | Rn_moyen_pixel = {np.nanmean(Rn):.1f} W/m²")
    LOGGER.info(f"      🔧 r_ah_s = {r_ah_s:.1f} s/m | r_ah_c = {r_ah_c:.1f} s/m")
    
    # =========================================================================
    # PHASE 3 : Décomposition de la Température (cœur du TTME)
    # =========================================================================
    
    # Pour chaque pixel : position relative dans le trapèze
    # a = T_rad - T_a (distance du pixel au bord froid)
    a = np.where(valid, lst_array - Ta, np.nan)
    
    # Température du bord chaud à ce fc : T_warm(fc) = T_s_max + beta_w * fc
    T_warm_at_fc = np.where(valid, T_s_max + beta_w * fc, np.nan)
    
    # a + b = T_warm(fc) - Ta (étendue totale à ce fc)
    a_plus_b = np.where(valid, T_warm_at_fc - Ta, np.nan)
    
    # Éviter les divisions par zéro et les valeurs hors trapèze
    a_plus_b = np.where(a_plus_b > 0.1, a_plus_b, np.nan)
    
    # Ratio de position (0 = bord froid, 1 = bord chaud)
    ratio = np.where(np.isfinite(a_plus_b), np.clip(a / a_plus_b, 0.0, 1.0), np.nan)
    
    # Pente de l'isoplèthe passant par ce pixel
    beta_i = np.where(np.isfinite(ratio), ratio * beta_w, np.nan)
    
    # Extraction des températures pures
    # T_s = T_rad - beta_i * fc  (on recule jusqu'à fc = 0)
    Ts = np.where(valid, lst_array - beta_i * fc, np.nan)
    
    # T_c = T_s + beta_i  (on avance jusqu'à fc = 1)
    Tc = np.where(valid, Ts + beta_i, np.nan)
    
    # Contraindre : Ts et Tc >= Ta (physiquement, la surface est plus chaude que l'air en journée)
    # On autorise un léger refroidissement (transpiration active) mais pas aberrant
    Ts = np.where(Ts < Ta - 5, np.nan, Ts)
    Tc = np.where(Tc < Ta - 5, np.nan, Tc)
    
    # =========================================================================
    # PHASE 4 : Paramétrisation Séparée des Flux (Géométrique TTME)
    # =========================================================================
    
    # L'énergie disponible pour chaque pôle pur
    G_pur_sol = cg_s * Rn
    G_pur_canopee = cg_c * Rn
    
    Rn_s_dispo = Rn - G_pur_sol
    Rn_c_dispo = Rn - G_pur_canopee
    
    # Calcul de la Chaleur Latente par interpolation linéaire dans le trapèze
    # Pour le sol : varie de LEs = Rn_s_dispo (si Ts = Ta) à LEs = 0 (si Ts = T_s_max)
    LEs = np.where(
        valid & (T_s_max > Ta),
        Rn_s_dispo * (T_s_max - Ts) / (T_s_max - Ta),
        0.0
    )
    
    # Pour la canopée : varie de LEc = Rn_c_dispo (si Tc = Ta) à LEc = 0 (si Tc = T_c_max)
    LEc = np.where(
        valid & (T_c_max > Ta),
        Rn_c_dispo * (T_c_max - Tc) / (T_c_max - Ta),
        0.0
    )
    
    # Par déduction, on obtient la Chaleur Sensible
    Hs = np.where(valid, Rn_s_dispo - LEs, np.nan)
    Hc = np.where(valid, Rn_c_dispo - LEc, np.nan)
    
    # =========================================================================
    # PHASE 5 : Synthèse - ET totale
    # =========================================================================
    
    # LE total (mosaïque pondérée par fc - Equation 1)
    LE = np.where(valid, fc * LEc + (1.0 - fc) * LEs, np.nan)
    
    # L'énergie disponible totale pour le pixel mixte (Rn_pixel - G_pixel)
    energie_dispo_pixel = np.where(valid, Rn - G, np.nan)
    
    # Contraindre thermodynamiquement LE :
    LE = np.clip(LE, 0.0, energie_dispo_pixel)
    
    # Conversion en ET (mm/h)
    ET_mm_h = np.where(valid, LE * 3600.0 / LAMBDA_V, np.nan)
    
    # Fraction évaporative globale du pixel
    EF = np.where((energie_dispo_pixel > 10) & valid, LE / energie_dispo_pixel, np.nan)
    EF = np.clip(EF, 0.0, 1.0)
    
    return {
        'fc': fc,
        'Ts': Ts,
        'Tc': Tc,
        'Hs': Hs,
        'Hc': Hc,
        'LEs': LEs,
        'LEc': LEc,
        'LE': LE,
        'ET_mm_h': ET_mm_h,
        'EF': EF,
        'G': G,
        'T_s_max': T_s_max,
        'T_c_max': T_c_max,
        'beta_w': beta_w,
        'transform': transform,
        'crs': crs,
    }


def save_et_tif(data_2d, transform, crs, output_path):
    """Sauvegarde un array 2D en GeoTIFF."""
    height, width = data_2d.shape
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=height, width=width, count=1,
        dtype='float32', crs=crs,
        transform=transform, nodata=np.nan
    ) as dst:
        dst.write(data_2d.astype(np.float32), 1)


def plot_trapezoid(fc, lst, ef, T_s_max, T_c_max, Ta, site, date_str, output_path):
    """Trace le graphique de l'espace trapézoïdal LST-fc."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Masque de données valides
    mask = np.isfinite(fc) & np.isfinite(lst) & np.isfinite(ef)
    if np.sum(mask) < 10:
        plt.close()
        return
    
    sc = ax.scatter(fc[mask].ravel(), lst[mask].ravel(), 
                    c=ef[mask].ravel(), cmap='RdYlBu', s=1, alpha=0.5, 
                    vmin=0, vmax=1)
    
    # Tracer les limites du trapèze
    fc_line = np.array([0, 1])
    # Bord chaud
    warm_line = np.array([T_s_max, T_c_max])
    ax.plot(fc_line, warm_line, 'r-', linewidth=2.5, label=f'Bord chaud (sec)')
    # Bord froid
    cold_line = np.array([Ta, Ta])
    ax.plot(fc_line, cold_line, 'b-', linewidth=2.5, label=f'Bord froid (humide)')
    
    plt.colorbar(sc, label='Fraction Évaporative (EF)', ax=ax)
    ax.set_xlabel('Fraction de Couverture Végétale (fc)', fontsize=12)
    ax.set_ylabel('Température de Surface LST (°C)', fontsize=12)
    ax.set_title(f'Espace Trapézoïdal TTME — {site} ({date_str})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main(source='icos', lst_source='dms'):
    source_label = source.upper()
    LOGGER.info("=" * 60)
    LOGGER.info(f"🌿 DÉMARRAGE DU CALCUL D'ÉVAPOTRANSPIRATION (TTME) — Source météo : {source_label}")
    LOGGER.info("=" * 60)
    
    BASE_TIF_DIR = OUTPUT_DIR
    resultats = []
    
    for site, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"🌍 SITE : {site}")
        
        # --- Recherche des fichiers TIF ---
        tif_folder = os.path.join(BASE_TIF_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data")
        if not os.path.exists(tif_folder):
            LOGGER.warning(f"   ⚠️ Dossier TIF absent : {tif_folder}")
            continue
        
        # Dossier de sortie pour les résultats ET (séparé par source)
        suffix_source = f"_{source.upper()}"
        suffix_lst = "_B10" if lst_source == 'b10' else ""
        et_output_dir = os.path.join(BASE_TIF_DIR, f"Serie_Temporelle_{site}", f"ET_TTME{suffix_source}{suffix_lst}")
        os.makedirs(et_output_dir, exist_ok=True)
        
        # Scanner les fichiers TIF et regrouper par date
        dict_dates = {}
        for f in os.listdir(tif_folder):
            if not f.endswith('.tif'):
                continue
            dt = extract_datetime_from_filename(f)
            if dt is None:
                continue
            if dt not in dict_dates:
                dict_dates[dt] = {}
            
            if "LST_Sharpened_DMS" in f and "Fusion" not in f:
                dict_dates[dt]['dms'] = os.path.join(tif_folder, f)
            elif "NDVI" in f:
                dict_dates[dt]['ndvi'] = os.path.join(tif_folder, f)
            elif "Thermique_B10" in f:
                dict_dates[dt]['b10'] = os.path.join(tif_folder, f)
            elif "Reflectance_B2" in f:
                dict_dates[dt]['b2'] = os.path.join(tif_folder, f)
            elif "Reflectance_B4" in f:
                dict_dates[dt]['b4'] = os.path.join(tif_folder, f)
            elif "Reflectance_B5" in f:
                dict_dates[dt]['b5'] = os.path.join(tif_folder, f)
            elif "Reflectance_B6" in f:
                dict_dates[dt]['b6'] = os.path.join(tif_folder, f)
            elif "Reflectance_B7" in f:
                dict_dates[dt]['b7'] = os.path.join(tif_folder, f)
        
        LOGGER.info(f"   📂 {len(dict_dates)} dates détectées.")
        
        nb_ok = 0
        nb_skip_meteo = 0
        nb_skip_data = 0
        
        for target_dt, paths in sorted(dict_dates.items()):
            date_str = target_dt.strftime("%Y-%m-%d")
            
            # Vérifier qu'on a les rasters nécessaires
            if lst_source == 'dms':
                path_lst = paths.get('dms') or paths.get('b10')
            elif lst_source == 'b10':
                path_lst = paths.get('b10')
            elif lst_source == 'dms_biais':
                path_lst = paths.get('dms')
            
            path_ndvi = paths.get('ndvi')
            path_b2 = paths.get('b2')
            path_b4 = paths.get('b4')
            path_b5 = paths.get('b5')
            path_b6 = paths.get('b6')
            path_b7 = paths.get('b7')
            
            if not path_lst or not path_ndvi or not path_b2 or not path_b4 or not path_b5 or not path_b6 or not path_b7:
                nb_skip_data += 1
                continue
            
            # --- Chargement de la météo ---
            meteo = load_meteo(site, target_dt, date_str, source=source)
            if meteo is None:
                nb_skip_meteo += 1
                continue
            
            Ta = meteo['Ta']
            u = meteo['u'] if pd.notna(meteo['u']) else 2.0  # Défaut 2 m/s
            Rn = meteo['Rn']
            G_meas = meteo.get('G', np.nan)
            
            if pd.isna(Ta) or pd.isna(Rn):
                nb_skip_meteo += 1
                continue
            
            # Filtrer les cas où Rn est trop faible (nuit ou coucher de soleil)
            if Rn < 50:
                LOGGER.info(f"   ⏭️  {date_str} : Rn trop faible ({Rn:.0f} W/m²), skip.")
                nb_skip_data += 1
                continue
            
            # --- Charger les rasters ---
            try:
                with rasterio.open(path_lst) as src_lst:
                    lst_array = src_lst.read(1).astype(np.float32)
                    transform = src_lst.transform
                    crs = src_lst.crs
                    # Conversion K → °C si nécessaire
                    lst_array = np.where(lst_array > 200, lst_array - 273.15, lst_array)
                    lst_array = np.where((lst_array < -50) | (lst_array > 80), np.nan, lst_array)
                
                with rasterio.open(path_ndvi) as src_ndvi:
                    ndvi_array = src_ndvi.read(1).astype(np.float32)
                    ndvi_array = np.where((ndvi_array < -1) | (ndvi_array > 1), np.nan, ndvi_array)
                    
                with rasterio.open(path_b2) as src_b2: b2_array = src_b2.read(1).astype(np.float32)
                with rasterio.open(path_b4) as src_b4: b4_array = src_b4.read(1).astype(np.float32)
                with rasterio.open(path_b5) as src_b5: b5_array = src_b5.read(1).astype(np.float32)
                with rasterio.open(path_b6) as src_b6: b6_array = src_b6.read(1).astype(np.float32)
                with rasterio.open(path_b7) as src_b7: b7_array = src_b7.read(1).astype(np.float32)
                
                # Aligner les dimensions si légèrement différentes (effet de bord du DMS)
                if lst_array.shape != ndvi_array.shape:
                    h_min = min(lst_array.shape[0], ndvi_array.shape[0], b2_array.shape[0])
                    w_min = min(lst_array.shape[1], ndvi_array.shape[1], b2_array.shape[1])
                    lst_array = lst_array[:h_min, :w_min]
                    ndvi_array = ndvi_array[:h_min, :w_min]
                    b2_array = b2_array[:h_min, :w_min]
                    b4_array = b4_array[:h_min, :w_min]
                    b5_array = b5_array[:h_min, :w_min]
                    b6_array = b6_array[:h_min, :w_min]
                    b7_array = b7_array[:h_min, :w_min]
                    LOGGER.info(f"   🔧 {date_str} : Recadrage rasters → ({h_min}, {w_min})")
                
                # Vérifier qu'il y a assez de pixels valides
                valid_pct = np.sum(np.isfinite(lst_array) & np.isfinite(ndvi_array)) / lst_array.size * 100
                if valid_pct < 20:
                    nb_skip_data += 1
                    continue
                    
            except Exception as e:
                LOGGER.error(f"   ❌ {date_str} : Erreur lecture raster : {e}")
                nb_skip_data += 1
                continue
            
            # --- Exécuter le modèle TTME ---
            LOGGER.info(f"   🌿 {date_str} : Calcul TTME (source météo: {meteo['source']})...")
            
            # Récupérer les paramètres du site
            site_params = SITE_TTME_PARAMS.get(site, SITE_TTME_PARAMS["default"])
            
            # Calcul du bilan radiatif spatialisé
            fc_array = compute_fc(ndvi_array, site_params)
            bandes_landsat = {'B2': b2_array, 'B4': b4_array, 'B5': b5_array, 'B6': b6_array, 'B7': b7_array}
            
            # On utilise le rayonnement de base pour tout le monde si ERA5
            R_s_down = meteo.get('R_s_down', np.nan)
            R_l_down = meteo.get('R_l_down', np.nan)
            
            if pd.isna(R_s_down) or pd.isna(R_l_down):
                LOGGER.warning(f"   ⚠️ Forçages radiatifs descendants manquants pour {date_str}, impossible de calculer Rn spatialisé.")
                nb_skip_meteo += 1
                continue
                
            # Correction du biais LST si demandé
            if lst_source == 'dms_biais':
                BIASES_LST = {
                    "Gebesee": -3.69,
                    "Selhausen": 1.02,
                    "Lonzee": 1.45,
                    "Voulundgaard": 3.78,
                    "Klingenberg": 2.24,
                    "Estrees-Mons": 5.45,
                    "Borgo Cioffi": 2.46,
                    "Grignon": 4.11,
                    "Lamasquere": 8.64,
                }
                if site in BIASES_LST:
                    bias = BIASES_LST[site]
                    lst_array = lst_array - bias
                    LOGGER.info(f"      🛰️ Biais corrigé pour la LST de {site} (Correction: {-bias:.2f}°C)")
            
            Rn_2d = calculer_bilan_radiatif(bandes_landsat, fc_array, lst_array, R_s_down, R_l_down)
            
            z_m_source = 10.0 if meteo['source'] in ['ERA5', 'ERA5_DS', 'ERA5_BIAIS'] else Z_M
            
            result = ttme_compute_et(
                lst_array, ndvi_array,
                Ta=Ta, u=u, Rn=Rn_2d, R_s_down=R_s_down, R_l_down=R_l_down, params=site_params, G_measured=G_meas,
                transform=transform, crs=crs, z_m=z_m_source
            )
            
            if result is None:
                continue
            
            # --- Sauvegarder les résultats ---
            prefix = f"{date_str}_{site}"
            
            # Sauvegarder ET en TIF
            et_tif_path = os.path.join(et_output_dir, f"{prefix}_ET_mm_h.tif")
            save_et_tif(result['ET_mm_h'], transform, crs, et_tif_path)
            
            # Sauvegarder LE en TIF
            le_tif_path = os.path.join(et_output_dir, f"{prefix}_LE_W_m2.tif")
            save_et_tif(result['LE'], transform, crs, le_tif_path)
            
            # Sauvegarder EF en TIF
            ef_tif_path = os.path.join(et_output_dir, f"{prefix}_EF.tif")
            save_et_tif(result['EF'], transform, crs, ef_tif_path)
            
            # Tracer le trapèze
            plot_path = os.path.join(et_output_dir, f"{prefix}_Trapeze_TTME.png")
            plot_trapezoid(
                result['fc'], lst_array, result['EF'],
                result['T_s_max'], result['T_c_max'], Ta,
                site, date_str, plot_path
            )
            
            # --- Extraction du pixel au point de la station ---
            try:
                transformer_proj = Transformer.from_crs("EPSG:4326", str(crs), always_xy=True)
                x_p, y_p = transformer_proj.transform(coords["lon"], coords["lat"])
                
                # Trouver l'index du pixel le plus proche
                col_idx = int((x_p - transform.c) / transform.a)
                row_idx = int((y_p - transform.f) / transform.e)
                
                h, w = lst_array.shape
                if 0 <= row_idx < h and 0 <= col_idx < w:
                    et_pixel = result['ET_mm_h'][row_idx, col_idx]
                    le_pixel = result['LE'][row_idx, col_idx]
                    ef_pixel = result['EF'][row_idx, col_idx]
                    ts_pixel = result['Ts'][row_idx, col_idx]
                    tc_pixel = result['Tc'][row_idx, col_idx]
                    fc_pixel = result['fc'][row_idx, col_idx]
                    lst_pixel = lst_array[row_idx, col_idx]
                else:
                    et_pixel = le_pixel = ef_pixel = ts_pixel = tc_pixel = fc_pixel = lst_pixel = np.nan
            except Exception:
                et_pixel = le_pixel = ef_pixel = ts_pixel = tc_pixel = fc_pixel = lst_pixel = np.nan
            
            # Statistiques sur la carte
            le_valid = result['LE'][np.isfinite(result['LE'])]
            et_valid = result['ET_mm_h'][np.isfinite(result['ET_mm_h'])]
            
            LOGGER.info(f"      ✅ ET carte : moy={np.nanmean(et_valid):.3f} mm/h | "
                        f"LE moy={np.nanmean(le_valid):.1f} W/m² | "
                        f"EF pixel={ef_pixel:.2f}" if pd.notna(ef_pixel) else 
                        f"      ✅ ET calculée (pixel station hors grille)")
            
            resultats.append({
                'Site': site,
                'Date': date_str,
                'Source_Meteo': meteo['source'],
                'Ta (°C)': round(Ta, 2),
                'u (m/s)': round(u, 2),
                'Rn (W/m²)': round(Rn, 1),
                'T_s_max (°C)': round(result['T_s_max'], 2),
                'T_c_max (°C)': round(result['T_c_max'], 2),
                'LST_pixel (°C)': round(lst_pixel, 2) if pd.notna(lst_pixel) else np.nan,
                'Ts_pixel (°C)': round(ts_pixel, 2) if pd.notna(ts_pixel) else np.nan,
                'Tc_pixel (°C)': round(tc_pixel, 2) if pd.notna(tc_pixel) else np.nan,
                'fc_pixel': round(fc_pixel, 3) if pd.notna(fc_pixel) else np.nan,
                'LE_pixel (W/m²)': round(le_pixel, 2) if pd.notna(le_pixel) else np.nan,
                'ET_pixel (mm/h)': round(et_pixel, 4) if pd.notna(et_pixel) else np.nan,
                'EF_pixel': round(ef_pixel, 3) if pd.notna(ef_pixel) else np.nan,
                'ET_moy_carte (mm/h)': round(np.nanmean(et_valid), 4) if len(et_valid) > 0 else np.nan,
                'LE_moy_carte (W/m²)': round(np.nanmean(le_valid), 1) if len(le_valid) > 0 else np.nan,
            })
            
            nb_ok += 1
        
        LOGGER.info(f"\n   📊 Bilan {site} : {nb_ok} dates traitées | "
                    f"{nb_skip_meteo} sans météo | {nb_skip_data} données insuffisantes")
    
    # --- Sauvegarde CSV de synthèse ---
    if resultats:
        df_final = pd.DataFrame(resultats)
        suffix_source = f"_{source.upper()}"
        suffix_lst = f"_{lst_source.upper()}" if lst_source != 'dms' else ""
        csv_path = os.path.join(OUTPUT_DIR, f"Resultats_ET_TTME{suffix_source}{suffix_lst}.csv")
        df_final.to_csv(csv_path, index=False)
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"💾 Résultats sauvegardés : {csv_path}")
        LOGGER.info(f"   {len(resultats)} calculs d'ET réalisés sur {df_final['Site'].nunique()} sites.")
        LOGGER.info(f"{'='*60}")
        
        # Résumé par site
        print(f"\n📊 RÉSUMÉ PAR SITE (source: {source.upper()}) :")
        for site in df_final['Site'].unique():
            df_site = df_final[df_final['Site'] == site]
            et_moy = df_site['ET_pixel (mm/h)'].mean()
            le_moy = df_site['LE_pixel (W/m²)'].mean()
            ef_moy = df_site['EF_pixel'].mean()
            n = len(df_site)
            print(f"   {site:20s} : N={n:3d} | ET={et_moy:.4f} mm/h | LE={le_moy:.1f} W/m² | EF={ef_moy:.2f}")
    else:
        LOGGER.warning("⚠️ Aucun résultat ET calculé. Vérifiez les données d'entrée.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcul d'ET via le modèle TTME")
    parser.add_argument(
        '--source', type=str, default='icos',
        choices=['icos', 'era5', 'era5_ds', 'era5_biais'],
        help="Source des données météo : 'icos', 'era5', 'era5_ds' ou 'era5_biais'"
    )
    parser.add_argument(
        '--lst', type=str, default='dms',
        choices=['dms', 'b10', 'dms_biais'],
        help="Source de la LST : 'dms' (sharpened) ou 'b10' (brute) ou 'dms_biais' (dms corrigée des biais)"
    )
    args = parser.parse_args()
    main(source=args.source, lst_source=args.lst)