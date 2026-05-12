"""
tsharp_fusion.py

TsHARP Fusion : utilise le NDVI Sentinel-2 (10m) harmonise
avec le thermique Landsat (100m) pour produire une carte LST a 10m.

Relation quadratique classique : LST = a * NDVI^2 + b * NDVI + c
"""

import os
import glob
import numpy as np
from scipy.ndimage import zoom
import rioxarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from config import SITES_PILOTES, LOGGER, TIME_MARGIN_MINUTES

DOSSIER_BASE = r"Outputs"

# =====================================================================
# COEFFICIENTS D'HARMONISATION SPECTRALE HLS v2.0 (Sentinel-2A -> OLI)
# rho_OLI = slope * rho_MSI + intercept
# Source : Claverie et al. (2018), NASA LP DAAC HLS User Guide
# =====================================================================
HLS_COEFFICIENTS = {
    "B02": {"slope": 0.9778, "intercept": -0.0040},  # Blue
    "B03": {"slope": 1.0053, "intercept": -0.0009},  # Green
    "B04": {"slope": 0.9765, "intercept":  0.0009},  # Red
    "B08": {"slope": 0.9983, "intercept": -0.0001},  # NIR
    "B11": {"slope": 1.0042, "intercept":  0.0001},  # SWIR1
}

def harmonize_s2_band(reflectance, band_name):
    """Applique la correction spectrale HLS pour harmoniser S2 vers OLI."""
    if band_name in HLS_COEFFICIENTS:
        coef = HLS_COEFFICIENTS[band_name]
        return coef["slope"] * reflectance + coef["intercept"]
    return reflectance

# =====================================================================
# FONCTIONS TSHARP CORE
# =====================================================================

def fit_tsharp(coarse_lst, coarse_ndvi, mask=None):
    """Fit quadratique LST = a*NDVI^2 + b*NDVI + c."""
    lst_flat = coarse_lst.ravel().astype(np.float64)
    ndvi_flat = coarse_ndvi.ravel().astype(np.float64)

    if mask is not None:
        mask_flat = mask.ravel().astype(bool)
        lst_flat = lst_flat[mask_flat]
        ndvi_flat = ndvi_flat[mask_flat]

    design = np.column_stack([ndvi_flat**2, ndvi_flat, np.ones_like(ndvi_flat)])
    coefficients, _, _, _ = np.linalg.lstsq(design, lst_flat, rcond=None)

    return coefficients.astype(np.float64)


def predict_tsharp(coarse_lst, coarse_ndvi, fine_ndvi, coefficients=None, mask=None):
    """Predit la LST fine resolution via TsHARP."""
    if coefficients is None:
        coefficients = fit_tsharp(coarse_lst, coarse_ndvi, mask=mask)

    a, b, c = coefficients

    coarse_predicted = (
        a * coarse_ndvi.astype(np.float64) ** 2
        + b * coarse_ndvi.astype(np.float64)
        + c
    )

    residual = coarse_lst.astype(np.float64) - coarse_predicted

    zoom_factors = (
        fine_ndvi.shape[0] / coarse_lst.shape[0],
        fine_ndvi.shape[1] / coarse_lst.shape[1],
    )
    residual_interp = zoom(residual, zoom_factors, order=1)

    fine_predicted = (
        a * fine_ndvi.astype(np.float64) ** 2
        + b * fine_ndvi.astype(np.float64)
        + c
        + residual_interp
    )

    return fine_predicted.astype(np.float32)


# =====================================================================
# FONCTIONS UTILITAIRES
# =====================================================================

def aggregate_block(matrice_2d, block_size):
    """Regroupe les pixels par blocs de NxN et calcule la moyenne."""
    h, w = matrice_2d.shape
    h_new = (h // block_size) * block_size
    w_new = (w // block_size) * block_size
    matrice_coupee = matrice_2d[:h_new, :w_new]
    return matrice_coupee.reshape(
        h_new // block_size, block_size, 
        w_new // block_size, block_size
    ).mean(axis=(1, 3))

def load_raster_as_2d(filepath):
    """Charge un fichier TIF et retourne sa matrice 2D."""
    ds = rioxarray.open_rasterio(filepath)
    array_2d = ds.values.squeeze()
    return array_2d

def parse_date_from_filename(date_str):
    """Convertit '2025-12-12_10h36' en objet datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d_%Hh%M")

def find_s2_match(nom_site, landsat_date_str, max_delta_minutes=30):
    """Cherche une image S2 quasi-simultanee."""
    s2_dir = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")
    
    if not os.path.exists(s2_dir):
        return None, None
    
    landsat_dt = parse_date_from_filename(landsat_date_str)
    fichiers_s2 = glob.glob(os.path.join(s2_dir, f"*_{nom_site}_S2_NDVI.tif"))
    
    meilleure_paire = None
    meilleur_delta = float('inf')
    
    for f in fichiers_s2:
        bn = os.path.basename(f)
        parts = bn.split('_')
        s2_date_str = f"{parts[0]}_{parts[1]}"
        
        try:
            s2_dt = parse_date_from_filename(s2_date_str)
        except ValueError:
            continue
        
        delta_minutes = abs((landsat_dt - s2_dt).total_seconds()) / 60
        
        if delta_minutes <= max_delta_minutes and delta_minutes < meilleur_delta:
            meilleur_delta = delta_minutes
            meilleure_paire = s2_date_str
    
    if meilleure_paire:
        return meilleure_paire, meilleur_delta
    return None, None


# =====================================================================
# FONCTION PRINCIPALE DE FUSION TSHARP
# =====================================================================

def process_tsharp_fusion(nom_site, landsat_date_str, s2_date_str, delta_minutes,
                          dossier_landsat, dossier_s2):
    """
    TsHARP Fusion : utilise le NDVI S2 (10m) avec le thermique Landsat.
    Apprentissage a ~100m, prediction a 10m.
    """
    LOGGER.info(f"\n   FUSION TsHARP {landsat_date_str} + S2:{s2_date_str} (delta={delta_minutes:.0f} min)")

    # 1. Charger le thermique Landsat
    fichier_thermique = os.path.join(dossier_landsat, f"{landsat_date_str}_{nom_site}_Thermique_B10.tif")
    fichier_sortie = os.path.join(dossier_landsat, f"{landsat_date_str}_{nom_site}_LST_Sharpened_TsHARP_Fusion.tif")
    fichier_comparaison = os.path.join(dossier_landsat, f"{landsat_date_str}_{nom_site}_Comparaison_TsHARP_Fusion.png")

    if not os.path.exists(fichier_thermique):
        LOGGER.warning(f"   Fichier thermique introuvable : {fichier_thermique}")
        return

    lst_landsat_2d = load_raster_as_2d(fichier_thermique)
    h_landsat, w_landsat = lst_landsat_2d.shape

    # 2. Charger les bandes S2 Red (B04) et NIR (B08) pour recalculer le NDVI harmonise
    # On utilise les TIF d'indices existants pour retrouver le chemin du dossier,
    # mais on a besoin des bandes brutes. Comme elles ne sont pas sauvegardees,
    # on charge le NDVI S2 existant puis on applique la correction HLS sur le NDVI.
    # 
    # Justification physique : NDVI = (NIR - Red) / (NIR + Red)
    # Apres harmonisation : NIR_h = a_nir * NIR + b_nir, Red_h = a_red * Red + b_red
    # Le NDVI harmonise est donc legerement different du NDVI brut.
    # Comme les slopes HLS sont proches de 1.0, l'effet est subtil mais mesurable.
    
    fichier_ndvi_s2 = os.path.join(dossier_s2, f"{s2_date_str}_{nom_site}_S2_NDVI.tif")
    
    if not os.path.exists(fichier_ndvi_s2):
        LOGGER.error(f"   NDVI S2 introuvable : {fichier_ndvi_s2}")
        return
    
    ndvi_s2_raw = load_raster_as_2d(fichier_ndvi_s2)
    
    # Approximation de l'harmonisation sur le NDVI :
    # On reconstruit Red et NIR approximatifs a partir du NDVI brut,
    # puis on applique les coefficients HLS et on recalcule le NDVI.
    # NDVI = (NIR - Red) / (NIR + Red)  =>  NIR = Red * (1 + NDVI) / (1 - NDVI)
    # On pose Red = 0.1 (valeur typique) pour reconstruire le ratio
    red_approx = 0.1 * np.ones_like(ndvi_s2_raw)
    ndvi_safe = np.clip(ndvi_s2_raw, -0.99, 0.99)  # Eviter division par 0
    nir_approx = red_approx * (1 + ndvi_safe) / (1 - ndvi_safe)
    
    # Harmonisation HLS
    red_h = harmonize_s2_band(red_approx, "B04")
    nir_h = harmonize_s2_band(nir_approx, "B08")
    
    # Recalcul du NDVI harmonise
    denom = nir_h + red_h
    ndvi_s2_10m = np.where(denom != 0, (nir_h - red_h) / denom, np.nan)
    ndvi_s2_10m = np.where(np.isnan(ndvi_s2_raw), np.nan, ndvi_s2_10m)  # Garder les NaN originaux
    
    h_s2, w_s2 = ndvi_s2_10m.shape
    LOGGER.info(f"   Grille Landsat : {h_landsat}x{w_landsat} (30m) | Grille S2 : {h_s2}x{w_s2} (10m)")
    LOGGER.info(f"   Harmonisation HLS appliquee sur le NDVI (Red slope={HLS_COEFFICIENTS['B04']['slope']}, NIR slope={HLS_COEFFICIENTS['B08']['slope']})")

    # 3. Degrader tout a ~100m pour l'apprentissage
    block_size_100m = 10  # 10 pixels S2 de 10m = 100m
    
    ndvi_100m = aggregate_block(ndvi_s2_10m, block_size_100m)
    h_100m, w_100m = ndvi_100m.shape
    
    # Re-echantillonner le thermique Landsat sur la grille 100m
    zoom_h = h_100m / h_landsat
    zoom_w = w_100m / w_landsat
    lst_100m_2d = zoom(lst_landsat_2d, (zoom_h, zoom_w), order=1)
    lst_100m_2d = lst_100m_2d[:h_100m, :w_100m]
    
    LOGGER.info(f"   Grille d'apprentissage a ~100m : {h_100m}x{w_100m}")

    # 4. TsHARP : apprentissage quadratique a ~100m, prediction a 10m
    mask_100m = np.isfinite(lst_100m_2d) & np.isfinite(ndvi_100m)

    try:
        lst_sharpened_10m = predict_tsharp(
            coarse_lst=np.nan_to_num(lst_100m_2d, nan=np.nanmean(lst_100m_2d)),
            coarse_ndvi=np.nan_to_num(ndvi_100m, nan=np.nanmean(ndvi_100m)),
            fine_ndvi=np.nan_to_num(ndvi_s2_10m, nan=np.nanmean(ndvi_s2_10m)),
            mask=mask_100m
        )
    except Exception as e:
        LOGGER.error(f"   Erreur lors de TsHARP Fusion : {e}")
        return

    # 5. Ajuster la taille et remettre les NaN
    lst_sharpened_10m = lst_sharpened_10m[:h_s2, :w_s2]
    masque_nan = np.isnan(ndvi_s2_10m)
    lst_sharpened_10m[masque_nan] = np.nan

    # 6. Sauvegarde TIF a 10m
    ds_base = rioxarray.open_rasterio(fichier_ndvi_s2)
    ds_out = ds_base.copy()
    ds_out.values = [lst_sharpened_10m]
    ds_out.rio.to_raster(fichier_sortie)
    LOGGER.info(f"   TIF HD 10m TsHARP Fusion sauvegarde : {fichier_sortie}")

    # 7. Sauvegarde visuelle PNG
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(lst_landsat_2d, cmap='magma', vmin=10, vmax=50)
    plt.title("Avant : Thermique Landsat 100m", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lst_sharpened_10m, cmap='magma', vmin=10, vmax=50)
    plt.title("Apres : TsHARP Fusion 10m", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.suptitle(f"{nom_site} - {landsat_date_str} (Landsat+S2 delta={delta_minutes:.0f}min)", fontsize=16)
    plt.tight_layout()
    plt.savefig(fichier_comparaison, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    LOGGER.info("========================================")
    LOGGER.info("DEMARRAGE DU TsHARP FUSION (Landsat Thermique + S2 NDVI)")
    LOGGER.info("========================================")

    for nom_site in SITES_PILOTES.keys():
        LOGGER.info(f"\n=== Traitement du site : {nom_site} ===")
        
        dossier_landsat = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}", "3_Indices", "TIF_Data")
        dossier_s2 = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")
        
        if not os.path.exists(dossier_landsat):
            LOGGER.info(f"   Pas de dossier Landsat pour {nom_site}. Skip.")
            continue
        
        if not os.path.exists(dossier_s2):
            LOGGER.info(f"   Pas de dossier S2 pour {nom_site}. Skip.")
            continue
            
        fichiers_thermiques = glob.glob(os.path.join(dossier_landsat, f"*_{nom_site}_Thermique_B10.tif"))
        
        nb_fusions = 0
        for chemin_fichier in fichiers_thermiques:
            nom_fichier = os.path.basename(chemin_fichier)
            parts = nom_fichier.split('_')
            landsat_date_str = f"{parts[0]}_{parts[1]}"
            
            s2_date_str, delta_minutes = find_s2_match(nom_site, landsat_date_str, TIME_MARGIN_MINUTES)
            
            if s2_date_str:
                process_tsharp_fusion(
                    nom_site, landsat_date_str, s2_date_str, delta_minutes,
                    dossier_landsat, dossier_s2
                )
                nb_fusions += 1
            else:
                LOGGER.info(f"   {landsat_date_str} : Pas de paire S2 (<{TIME_MARGIN_MINUTES} min). TsHARP classique uniquement.")
        
        LOGGER.info(f"   {nb_fusions} fusion(s) realisee(s) pour {nom_site}.")

    LOGGER.info("\nTraitement TsHARP Fusion termine pour tous les sites !")

if __name__ == "__main__":
    main()
