"""
Extraction/ERA5/extraction_ERA5.py
===================================
Extraction des données météorologiques ERA5 (réanalyse ECMWF) pour tous les sites pilotes.
Sauvegarde en CSV avec le même format de colonnes que les données ICOS/NOAA,
pour pouvoir les utiliser directement dans le calcul d'ET (TTME).

Prérequis :
  1. Créer un compte sur https://cds.climate.copernicus.eu/
  2. Accepter les conditions d'utilisation du dataset ERA5
  3. Récupérer ta clé API dans ton profil CDS
  4. Créer le fichier ~/.cdsapirc (ou %USERPROFILE%\\.cdsapirc sur Windows) :
        url: https://cds.climate.copernicus.eu/api
        key: <ta-clé-api>

Variables extraites :
  - Ta : Température de l'air à 2m (°C)
  - WS : Vitesse du vent à 10m → ajustée à 2m (m/s)
  - RH : Humidité relative (%) - calculée depuis T et Td
  - SW_IN : Rayonnement shortwave descendant (W/m²)
  - LW_IN : Rayonnement longwave descendant (W/m²)
  - Rn : Rayonnement net (W/m²)
  - PA : Pression atmosphérique (kPa)
  - LST_Calculee : Température de peau (skin temperature) ERA5 (°C)
"""

import os
import sys
import numpy as np
import pandas as pd

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

from config import LOGGER, SITES_PILOTES, TIME_OF_INTEREST


# =============================================================================
# CONSTANTES
# =============================================================================
DOSSIER_SORTIE = "Outputs_ERA5"

# Variables ERA5 à télécharger (Single Levels - Reanalysis)
ERA5_VARIABLES = [
    '2m_temperature',                       # T2m en Kelvin
    '2m_dewpoint_temperature',              # Td2m en Kelvin (pour calculer RH)
    '10m_u_component_of_wind',              # u10 en m/s
    '10m_v_component_of_wind',              # v10 en m/s
    'surface_solar_radiation_downward_clear_sky',    # SW↓_clear en J/m² (cumulatif sur 1h)
    'surface_thermal_radiation_downward_clear_sky',  # LW↓_clear en J/m² (cumulatif sur 1h)
    'surface_net_solar_radiation',          # SW_net en J/m² (cumulatif)
    'surface_net_thermal_radiation',        # LW_net en J/m² (cumulatif)
    'surface_pressure',                     # Pression en Pa
    'skin_temperature',                     # Température de surface en K
]

# Heures d'intérêt (autour du passage Landsat ~10h30 heure solaire)
HEURES_INTERET = [
    '08:00', '09:00', '10:00', '11:00', '12:00', '13:00',
]


# =============================================================================
# FONCTIONS DE CONVERSION PHYSIQUE
# =============================================================================

def kelvin_to_celsius(t_k):
    """Convertit Kelvin en Celsius."""
    return t_k - 273.15


def compute_rh_from_t_td(t2m_k, td2m_k):
    """
    Calcule l'humidité relative (%) à partir de T et Td (en Kelvin).
    Formule de Magnus-Tetens.
    """
    t = t2m_k - 273.15  # °C
    td = td2m_k - 273.15  # °C
    
    # Constantes de Magnus
    a = 17.625
    b = 243.04  # °C
    
    # Pression de vapeur saturante et réelle
    gamma_t = (a * t) / (b + t)
    gamma_td = (a * td) / (b + td)
    
    rh = 100.0 * np.exp(gamma_td - gamma_t)
    return np.clip(rh, 0, 100)


def wind_10m_to_2m(ws_10m):
    """
    Ajuste la vitesse du vent de 10m à 2m via le profil logarithmique.
    u(z) = u(z_ref) * ln(z/z0) / ln(z_ref/z0)
    Avec z0 ≈ 0.03m (terrain agricole moyen).
    """
    z0 = 0.03  # m
    return ws_10m * np.log(2.0 / z0) / np.log(10.0 / z0)


def cumulative_to_instantaneous(values_j_m2, dt_seconds=3600):
    """
    Convertit les valeurs cumulatives ERA5 (J/m²) en flux instantanés (W/m²).
    Pour les données horaires ERA5 reanalysis, la valeur est accumulée sur 1h.
    """
    return values_j_m2 / dt_seconds


# =============================================================================
# TÉLÉCHARGEMENT VIA CDS API
# =============================================================================

def download_era5_for_site(site, lon, lat, start_date, end_date, output_nc):
    """
    Télécharge les données ERA5 pour un site donné via l'API CDS.
    Découpe la requête mois par mois pour rester dans les limites CDS.
    Fusionne les fichiers mensuels en un seul NetCDF.
    """
    import cdsapi
    import xarray as xr
    
    # Bounding box : ±0.25° autour du site (juste le point de grille ERA5 le plus proche)
    margin = 0.25
    area = [lat + margin, lon - margin, lat - margin, lon + margin]  # [N, W, S, E]
    
    LOGGER.info(f"   📥 Téléchargement ERA5 pour {site} ({lon:.3f}, {lat:.3f})...")
    LOGGER.info(f"      Période : {start_date.date()} → {end_date.date()}")
    
    client = cdsapi.Client()
    
    # Générer la liste des mois à télécharger
    months_to_download = pd.date_range(start_date, end_date, freq='MS')  # début de chaque mois
    
    tmp_dir = os.path.dirname(output_nc)
    nc_parts = []
    
    for month_start in months_to_download:
        year = month_start.year
        month = month_start.month
        
        # Fichier temporaire pour ce mois
        nc_month = os.path.join(tmp_dir, f"era5_{site}_{year}_{month:02d}.nc")
        
        if os.path.exists(nc_month):
            LOGGER.info(f"      ✅ {year}-{month:02d} déjà téléchargé.")
            nc_parts.append(nc_month)
            continue
        
        LOGGER.info(f"      📥 {year}-{month:02d}...")
        
        days = [f"{d:02d}" for d in range(1, 32)]
        
        try:
            client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': ERA5_VARIABLES,
                    'year': str(year),
                    'month': f"{month:02d}",
                    'day': days,
                    'time': HEURES_INTERET,
                    'area': area,
                    'format': 'netcdf',
                    'download_format': 'unarchived',
                },
                nc_month
            )
            
            # Nouveau CDS-API peut retourner un ZIP même si format=netcdf est demandé
            import zipfile
            import uuid
            if zipfile.is_zipfile(nc_month):
                extracted_paths = []
                with zipfile.ZipFile(nc_month, 'r') as zip_ref:
                    extracted_files = zip_ref.namelist()
                    for f in extracted_files:
                        path = zip_ref.extract(f, tmp_dir)
                        # S'assurer d'un nom unique
                        uid = str(uuid.uuid4())[:8]
                        new_path = os.path.join(tmp_dir, f"{year}_{month:02d}_{uid}_{f}")
                        os.replace(path, new_path)
                        extracted_paths.append(new_path)
                
                # Remplacer le zip par un NetCDF fusionné
                os.remove(nc_month)
                
                ds_list_vars = [xr.open_dataset(p, engine='netcdf4') for p in extracted_paths]
                ds_merged_vars = xr.merge(ds_list_vars, compat='override')
                ds_merged_vars.to_netcdf(nc_month)
                
                for ds in ds_list_vars:
                    ds.close()
                ds_merged_vars.close()
                for p in extracted_paths:
                    try:
                        os.remove(p)
                    except:
                        pass
                
            nc_parts.append(nc_month)
        except Exception as e:
            LOGGER.warning(f"      ⚠️ Échec {year}-{month:02d} : {e}")
            continue
    
    if not nc_parts:
        raise RuntimeError(f"Aucun mois téléchargé pour {site}")
    
    # Fusionner tous les fichiers mensuels en un seul NetCDF
    LOGGER.info(f"   🔄 Fusion de {len(nc_parts)} fichiers mensuels...")
    ds_list = [xr.open_dataset(f, engine='netcdf4') for f in nc_parts]
    ds_merged = xr.concat(ds_list, dim='valid_time')
    ds_merged.to_netcdf(output_nc)
    for ds in ds_list:
        ds.close()
    
    LOGGER.info(f"   ✅ Fichier NetCDF fusionné : {output_nc}")
    return output_nc


# =============================================================================
# TRAITEMENT DU NETCDF → CSV
# =============================================================================

def process_era5_netcdf(nc_path, site, lon, lat):
    """
    Lit le fichier NetCDF ERA5, extrait le point le plus proche du site,
    et retourne un DataFrame avec les mêmes colonnes que les CSV ICOS.
    """
    import xarray as xr
    
    LOGGER.info(f"   🔄 Traitement du NetCDF pour {site}...")
    
    ds = xr.open_dataset(nc_path)
    
    # Extraction du point le plus proche
    ds_point = ds.sel(
        longitude=lon, latitude=lat, method='nearest'
    )
    
    # Convertir en DataFrame pandas
    df = ds_point.to_dataframe().reset_index()
    
    # Renommer la colonne temporelle
    time_col = 'valid_time' if 'valid_time' in df.columns else 'time'
    df.rename(columns={time_col: 'TIMESTAMP'}, inplace=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']).dt.tz_localize(None)
    df = df.sort_values('TIMESTAMP').reset_index(drop=True)
    
    # --- Conversions physiques ---
    
    # 1. Température de l'air (K → °C)
    df['TA_Consolide'] = kelvin_to_celsius(df['t2m'])
    
    # 2. Vitesse du vent (10m → 2m)
    df['WS_10m'] = np.sqrt(df['u10']**2 + df['v10']**2)
    df['WS_Consolide'] = wind_10m_to_2m(df['WS_10m'])
    
    # 3. Humidité relative (depuis T et Td)
    df['RH_Consolide'] = compute_rh_from_t_td(df['t2m'], df['d2m'])
    
    # 4. Rayonnements (J/m² cumulatif → W/m² instantané)
    df['SW_IN_Consolide'] = cumulative_to_instantaneous(df['ssrdc'])
    df['LW_IN_Consolide'] = cumulative_to_instantaneous(df['strdc'])
    
    # Rayonnement net
    sw_net = cumulative_to_instantaneous(df['ssr'])
    lw_net = cumulative_to_instantaneous(df['str'])
    df['Rn_Consolide'] = sw_net + lw_net
    
    # SW_OUT et LW_OUT (recalculés)
    df['SW_OUT_Consolide'] = df['SW_IN_Consolide'] - sw_net
    df['LW_OUT_Consolide'] = df['LW_IN_Consolide'] - lw_net
    
    # 5. Pression atmosphérique (Pa → kPa)
    df['PA_Consolide'] = df['sp'] / 1000.0
    
    # 6. LST (skin temperature, K → °C)
    df['LST_Calculee'] = kelvin_to_celsius(df['skt'])
    
    # Sélectionner les colonnes finales (même format que ICOS)
    colonnes_finales = [
        'TIMESTAMP',
        'LW_IN_Consolide', 'LW_OUT_Consolide', 'LST_Calculee',
        'TA_Consolide', 'WS_Consolide', 'RH_Consolide',
        'SW_IN_Consolide', 'SW_OUT_Consolide',
        'PA_Consolide', 'Rn_Consolide',
    ]
    
    df_final = df[colonnes_finales].copy()
    df_final = df_final.set_index('TIMESTAMP')
    
    # Nettoyage
    df_final.replace([-9999.0, -999.0], np.nan, inplace=True)
    
    ds.close()
    
    LOGGER.info(f"   📊 {len(df_final)} mesures horaires extraites pour {site}")
    LOGGER.info(f"      Ta : [{df_final['TA_Consolide'].min():.1f}, {df_final['TA_Consolide'].max():.1f}] °C")
    LOGGER.info(f"      WS : [{df_final['WS_Consolide'].min():.1f}, {df_final['WS_Consolide'].max():.1f}] m/s")
    LOGGER.info(f"      Rn : [{df_final['Rn_Consolide'].min():.1f}, {df_final['Rn_Consolide'].max():.1f}] W/m²")
    
    return df_final


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    LOGGER.info("=" * 60)
    LOGGER.info("🌐 EXTRACTION ERA5 — Données Météorologiques de Réanalyse")
    LOGGER.info("=" * 60)
    
    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    tmp_dir = os.path.join(DOSSIER_SORTIE, "_tmp_nc")
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Période d'intérêt
    start_str, end_str = TIME_OF_INTEREST.split('/')
    start_date = pd.to_datetime(start_str)
    end_date = pd.to_datetime(end_str)
    
    for site, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n{'='*50}")
        LOGGER.info(f"🌍 Site : {site} (lon={coords['lon']:.3f}, lat={coords['lat']:.3f})")
        
        csv_path = os.path.join(DOSSIER_SORTIE, f"donnees_era5_{site}.csv")
        
        # Vérifier si le CSV existe déjà
        if os.path.exists(csv_path):
            LOGGER.info(f"   ✅ CSV ERA5 déjà existant : {csv_path}")
            LOGGER.info(f"   ℹ️  Supprime le fichier pour forcer le re-téléchargement.")
            continue
        
        # 1. Télécharger le NetCDF
        nc_path = os.path.join(tmp_dir, f"era5_{site}.nc")
        
        try:
            if not os.path.exists(nc_path):
                download_era5_for_site(
                    site, coords['lon'], coords['lat'],
                    start_date, end_date, nc_path
                )
            
            # 2. Traiter le NetCDF → DataFrame
            df = process_era5_netcdf(nc_path, site, coords['lon'], coords['lat'])
            
            # 3. Sauvegarder en CSV
            df.to_csv(csv_path)
            LOGGER.info(f"   💾 CSV ERA5 sauvegardé : {csv_path}")
            
        except Exception as e:
            LOGGER.error(f"   ❌ Erreur pour {site} : {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            continue
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("✅ EXTRACTION ERA5 TERMINÉE.")
    LOGGER.info(f"{'='*60}")


if __name__ == "__main__":
    main()
