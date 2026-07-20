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

# --- CONFIGURATION PROXY FOURNIE PAR L'UTILISATEUR ---
# --- AUCUN PROXY FORCÉ ---
# L'utilisateur a nettoyé ses variables d'environnement.
# -------------------------------------------------------

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path to import config from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# --- DISABLE PROXY SETTINGS & SSL VERIFICATION ---
# Clear any proxy environment variables that might interfere with CDS API
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

# Disable SSL verification for the entire process
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''

# Suppress SSL warnings for requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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
    import requests
    
    # Bounding box : ±0.25° autour du site (juste le point de grille ERA5 le plus proche)
    margin = 0.25
    area = [lat + margin, lon - margin, lat - margin, lon + margin]  # [N, W, S, E]
    
    LOGGER.info(f"   📥 Téléchargement ERA5 pour {site} ({lon:.3f}, {lat:.3f})...")
    LOGGER.info(f"      Période : {start_date.date()} → {end_date.date()}")
    
    # Create session without proxy and with SSL verification disabled
    session = requests.Session()
    session.proxies = {}  # Disable all proxies
    session.verify = False  # Disable SSL verification
    client = cdsapi.Client(session=session)
    
    # Générer la liste des mois à télécharger
    months_to_download = pd.date_range(start_date, end_date, freq='MS')  # début de chaque mois
    
    tmp_dir = os.path.dirname(output_nc)
    nc_parts = []
    
    month_count = 0
    for month_start in months_to_download:
        month_count += 1
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
        
        import time
        max_retries = 5
        for attempt in range(max_retries):
            try:
                client.retrieve(
                    'reanalysis-era5-land',
                    {
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
                break  # Succès, on sort de la boucle de retry
            except Exception as e:
                error_str = str(e)
                # SSL errors need longer wait times
                is_ssl_error = 'SSL' in error_str or 'certificate' in error_str.lower()
                wait_time = 60 if is_ssl_error else 10
                
                LOGGER.warning(f"      ⚠️ Échec {year}-{month:02d} (tentative {attempt+1}/{max_retries}) : {e}")
                if attempt < max_retries - 1:
                    LOGGER.info(f"      ⏳ Attente {wait_time}s avant nouvelle tentative...")
                    time.sleep(wait_time)
                else:
                    LOGGER.error(f"      ❌ Échec définitif pour {year}-{month:02d}.")
    
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
# EXTRACTION 2D SPATIALE DE Ta POUR LE DOWNSCALING
# =============================================================================

def download_era5_ta_2d(site, lon, lat, target_datetime, output_nc=None):
    """
    Télécharge un champ 2D de température de l'air (2m) ERA5-Land
    pour une date/heure spécifique, couvrant une emprise spatiale large
    autour du site (~30-40 km) nécessaire au stepwise downscaling.

    Parameters
    ----------
    site : str
        Nom du site.
    lon, lat : float
        Coordonnées du centre du site (WGS84).
    target_datetime : pd.Timestamp
        Date et heure cible (ex: passage Landsat).
    output_nc : str, optional
        Chemin de sortie du fichier NetCDF. Si None, généré automatiquement.

    Returns
    -------
    xr.DataArray
        Champ 2D de Ta en °C, géoréférencé en lat/lon (EPSG:4326).
    """
    import cdsapi
    import xarray as xr

    # Marge de ±0.3° ≈ 4 pixels ERA5-Land (9km) dans chaque direction ≈ 36 km
    margin = 0.30
    area = [lat + margin, lon - margin, lat - margin, lon + margin]  # [N, W, S, E]

    # Arrondir l'heure à l'heure ERA5 la plus proche
    hour_rounded = target_datetime.round('h').strftime('%H:%M')
    date_str = target_datetime.strftime('%Y-%m-%d')

    if output_nc is None:
        os.makedirs(os.path.join(DOSSIER_SORTIE, "_tmp_nc_2d"), exist_ok=True)
        output_nc = os.path.join(
            DOSSIER_SORTIE,
            "_tmp_nc_2d",
            f"era5_ta2d_{site}_{date_str}_{hour_rounded.replace(':', 'h')}.nc"
        )

    if os.path.exists(output_nc):
        LOGGER.info(f"   ✅ Champ 2D Ta déjà téléchargé : {output_nc}")
        ds = xr.open_dataset(output_nc)
        ta_var = 't2m' if 't2m' in ds else list(ds.data_vars)[0]
        ta_2d = ds[ta_var].isel(valid_time=0) if 'valid_time' in ds[ta_var].dims else ds[ta_var]
        import numpy as np
        ta_celsius = ta_2d - 273.15 if np.nanmean(ta_2d.values) > 100 else ta_2d
        ds.close()
        return ta_celsius

    LOGGER.info(f"   📥 Téléchargement du champ 2D de Ta ERA5-Land pour {site}...")
    LOGGER.info(f"      Date/heure : {date_str} {hour_rounded}")
    LOGGER.info(f"      Emprise : [{area[2]:.2f}°N, {area[1]:.2f}°E] → [{area[0]:.2f}°N, {area[3]:.2f}°E]")

    import requests
    
    # Create session without proxy and with SSL verification disabled
    session = requests.Session()
    session.proxies = {}  # Disable all proxies
    session.verify = False  # Disable SSL verification
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            client = cdsapi.Client(session=session)
            client.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': ['2m_temperature'],
                    'year': target_datetime.strftime('%Y'),
                    'month': target_datetime.strftime('%m'),
                    'day': target_datetime.strftime('%d'),
                    'time': [hour_rounded],
                    'area': area,
                    'format': 'netcdf',
                    'download_format': 'unarchived',
                },
                output_nc
            )

            # Gérer le cas ZIP (comme dans download_era5_for_site)
            import zipfile
            if zipfile.is_zipfile(output_nc):
                tmp_dir = os.path.dirname(output_nc)
                with zipfile.ZipFile(output_nc, 'r') as zip_ref:
                    extracted = zip_ref.namelist()
                    path = zip_ref.extract(extracted[0], tmp_dir)
                os.remove(output_nc)
                os.rename(path, output_nc)

            ds = xr.open_dataset(output_nc)
            ta_var = 't2m' if 't2m' in ds else list(ds.data_vars)[0]

            # Sélectionner le premier pas de temps si présent
            ta_2d = ds[ta_var]
            if 'valid_time' in ta_2d.dims:
                ta_2d = ta_2d.isel(valid_time=0)
            elif 'time' in ta_2d.dims:
                ta_2d = ta_2d.isel(time=0)

            # Kelvin → Celsius
            import numpy as np
            ta_celsius = ta_2d - 273.15 if np.nanmean(ta_2d.values) > 100 else ta_2d

            ds.close()
            LOGGER.info(f"   ✅ Champ 2D Ta : {ta_celsius.shape}, "
                        f"[{float(ta_celsius.min()):.1f}°C, {float(ta_celsius.max()):.1f}°C]")
            return ta_celsius
            
        except Exception as e:
            import time
            error_str = str(e)
            is_ssl_error = 'SSL' in error_str or 'certificate' in error_str.lower()
            wait_time = 60 if is_ssl_error else 10
            
            LOGGER.warning(f"   ⚠️ Échec tentative {attempt+1}/{max_retries} : {e}")
            if attempt < max_retries - 1:
                LOGGER.info(f"   ⏳ Attente {wait_time}s avant nouvelle tentative...")
                time.sleep(wait_time)
            else:
                LOGGER.error(f"   ❌ Erreur définitive téléchargement 2D Ta : {e}")
                import traceback
                LOGGER.error(traceback.format_exc())
                return None


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
        
        # 1. Télécharger le NetCDF
        nc_path = os.path.join(tmp_dir, f"era5_{site}_{start_date.year}.nc")
        
        try:
            if not os.path.exists(nc_path):
                download_era5_for_site(
                    site, coords['lon'], coords['lat'],
                    start_date, end_date, nc_path
                )
            
            # 2. Traiter le NetCDF → DataFrame
            df = process_era5_netcdf(nc_path, site, coords['lon'], coords['lat'])
            
            # 3. Sauvegarder en CSV
            if os.path.exists(csv_path):
                old_df = pd.read_csv(csv_path)
                first_col = old_df.columns[0]
                if first_col != 'TIMESTAMP':
                    old_df.rename(columns={first_col: 'TIMESTAMP'}, inplace=True)
                old_df['TIMESTAMP'] = pd.to_datetime(old_df['TIMESTAMP']).dt.tz_localize(None)
                df_combined = pd.concat([old_df, df]).drop_duplicates(subset=['TIMESTAMP']).sort_values('TIMESTAMP')
                df_combined.to_csv(csv_path, index=False)
                LOGGER.info(f"   ✅ Données ajoutées au CSV existant : {csv_path}")
            else:
                df.to_csv(csv_path, index=False)
                LOGGER.info(f"   💾 Nouveau CSV ERA5 sauvegardé : {csv_path}")
            
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
