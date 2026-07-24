import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import ee
from config import OUTPUT_DIR, SITES_PILOTES

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration du Logger
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "extraction_ERA5_GEE.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
LOGGER = logging.getLogger(__name__)

# =============================================================================
# FONCTIONS DE CONVERSION PHYSIQUE
# =============================================================================

def kelvin_to_celsius(t_k):
    return t_k - 273.15

def compute_rh_from_t_td(t_k, td_k):
    """
    Calcule l'Humidité Relative (%) à partir de T (température) et Td (point de rosée) en Kelvin.
    Formule de Magnus-Tetens.
    """
    t_c = kelvin_to_celsius(t_k)
    td_c = kelvin_to_celsius(td_k)
    
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    
    rh = (e / es) * 100.0
    return np.clip(rh, 0, 100)

def wind_10m_to_2m(ws_10m):
    """
    Ajuste la vitesse du vent de 10m à 2m via le profil logarithmique.
    """
    z0 = 0.03  # m
    return ws_10m * np.log(2.0 / z0) / np.log(10.0 / z0)

def cumulative_to_instantaneous(values_j_m2, dt_seconds=3600):
    """
    Convertit les valeurs cumulatives ERA5 (J/m²) en flux instantanés (W/m²).
    """
    return values_j_m2 / dt_seconds

# =============================================================================
# EXTRACTION GOOGLE EARTH ENGINE
# =============================================================================

def extract_gee_era5(site, lon, lat, start_year, end_year):
    """
    Extrait la série temporelle ERA5-Land depuis Google Earth Engine pour un point précis.
    Effectue l'extraction année par année pour éviter la limite de 5000 éléments de getInfo().
    """
    LOGGER.info(f"   📥 Téléchargement ERA5 via GEE pour {site} ({lon}, {lat})...")
    
    point = ee.Geometry.Point([lon, lat])
    records = []
    
    for year in range(start_year, end_year + 1):
        LOGGER.info(f"      📅 Extraction pour l'année {year}...")
        
        # Filtre sur l'année et UNIQUEMENT sur les heures d'intérêt (8h à 13h)
        collection = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
            .filterBounds(point) \
            .filterDate(f"{year}-01-01", f"{year+1}-01-01") \
            .filter(ee.Filter.calendarRange(8, 13, 'hour')) \
            .select([
                'temperature_2m', 
                'dewpoint_temperature_2m', 
                'u_component_of_wind_10m', 
                'v_component_of_wind_10m',
                'surface_solar_radiation_downwards_hourly', 
                'surface_thermal_radiation_downwards_hourly',
                'surface_net_solar_radiation_hourly', 
                'surface_net_thermal_radiation_hourly',
                'surface_pressure'
            ])
        
        def extract_point(image):
            val = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=11132,
                crs='EPSG:4326'
            )
            return ee.Feature(None, val).set('system:time_start', image.get('system:time_start'))
        
        timeseries = collection.map(extract_point)
        info = timeseries.getInfo()
        
        if 'features' in info:
            for f in info['features']:
                props = f['properties']
                timestamp_ms = props.get('system:time_start')
                if timestamp_ms is None:
                    continue
                    
                dt = pd.to_datetime(timestamp_ms, unit='ms')
                
                records.append({
                    'TIMESTAMP': dt,
                    't2m': props.get('temperature_2m', np.nan),
                    'd2m': props.get('dewpoint_temperature_2m', np.nan),
                    'u10': props.get('u_component_of_wind_10m', np.nan),
                    'v10': props.get('v_component_of_wind_10m', np.nan),
                    'ssrd': props.get('surface_solar_radiation_downwards_hourly', np.nan),
                    'strd': props.get('surface_thermal_radiation_downwards_hourly', np.nan),
                    'ssr': props.get('surface_net_solar_radiation_hourly', np.nan),
                    'str_net': props.get('surface_net_thermal_radiation_hourly', np.nan),
                    'sp': props.get('surface_pressure', np.nan)
                })
    
    if not records:
        raise RuntimeError(f"Aucune donnée GEE trouvée pour {site}.")
        
    df = pd.DataFrame(records)
    
    # Conversions
    df['TA_Consolide'] = kelvin_to_celsius(df['t2m'])
    df['WS_10m'] = np.sqrt(df['u10']**2 + df['v10']**2)
    df['WS_Consolide'] = wind_10m_to_2m(df['WS_10m'])
    df['RH_Consolide'] = compute_rh_from_t_td(df['t2m'], df['d2m'])
    
    # Rayonnements GEE ERA5-Land (Déjà en J/m2)
    df['SW_IN_Consolide'] = cumulative_to_instantaneous(df['ssrd'])
    df['LW_IN_Consolide'] = cumulative_to_instantaneous(df['strd'])
    
    sw_net = cumulative_to_instantaneous(df['ssr'])
    lw_net = cumulative_to_instantaneous(df['str_net'])
    df['Rn_Consolide'] = sw_net + lw_net
    
    df['SW_OUT_Consolide'] = df['SW_IN_Consolide'] - sw_net
    df['LW_OUT_Consolide'] = df['LW_IN_Consolide'] - lw_net
    
    df['PA_Consolide'] = df['sp'] / 1000.0
    
    return df

def download_era5_ta_2d(site, lon, lat, target_datetime, output_nc=None):
    """
    Télécharge un champ 2D de température de l'air (2m) ERA5-Land
    pour une date/heure spécifique, couvrant une emprise spatiale large
    autour du site (~30-40 km) nécessaire au stepwise downscaling.
    """
    import cdsapi
    import xarray as xr
    import os
    import requests

    # Marge de ±0.3° ≈ 4 pixels ERA5-Land (9km) dans chaque direction ≈ 36 km
    margin = 0.30
    area = [lat + margin, lon - margin, lat - margin, lon + margin]  # [N, W, S, E]

    # Arrondir l'heure à l'heure ERA5 la plus proche
    hour_rounded = target_datetime.round('h').strftime('%H:%M')
    date_str = target_datetime.strftime('%Y-%m-%d')
    
    DOSSIER_SORTIE = "Outputs_ERA5"

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

    session = requests.Session()
    session.proxies = {}  
    session.verify = False  

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

            ta_2d = ds[ta_var]
            if 'valid_time' in ta_2d.dims:
                ta_2d = ta_2d.isel(valid_time=0)
            elif 'time' in ta_2d.dims:
                ta_2d = ta_2d.isel(time=0)

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
                return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    LOGGER.info("=== DÉBUT EXTRACTION ERA5 VIA GOOGLE EARTH ENGINE ===")
    
    # 1. Initialiser l'API GEE
    try:
        ee.Initialize(project='et-atos')
    except Exception as e:
        LOGGER.error("Erreur GEE Initialize. Avez-vous exécuté 'earthengine authenticate' ?")
        LOGGER.error(e)
        return
        
    out_dir_final = "Outputs_ERA5"
    os.makedirs(out_dir_final, exist_ok=True)
    
    for site, info in SITES_PILOTES.items():
        lat = info['lat']
        lon = info['lon']
        LOGGER.info(f"🌍 Site : {site} (lon={lon}, lat={lat})")
        
        csv_out = os.path.join(out_dir_final, f"donnees_era5_{site}.csv")
        # if os.path.exists(csv_out):
        #     LOGGER.info(f"   ✅ CSV déjà existant pour {site}. Ignoré.")
        #     continue
            
        start_year = 2021
        end_year = 2024
        
        try:
            df = extract_gee_era5(site, lon, lat, start_year, end_year)
            
            # Formater les colonnes
            cols_export = [
                'TIMESTAMP', 'TA_Consolide', 'WS_Consolide', 'RH_Consolide',
                'Rn_Consolide', 'SW_IN_Consolide', 'LW_IN_Consolide', 
                'SW_OUT_Consolide', 'LW_OUT_Consolide', 'PA_Consolide'
            ]
            
            df_export = df[cols_export].copy()
            df_export.rename(columns={
                'TA_Consolide': 'Ta (°C)',
                'WS_Consolide': 'u (m/s)',
                'RH_Consolide': 'RH (%)',
                'Rn_Consolide': 'Rn (W/m²)',
                'SW_IN_Consolide': 'R_s_down (W/m²)',
                'LW_IN_Consolide': 'R_l_down (W/m²)',
                'SW_OUT_Consolide': 'R_s_up (W/m²)',
                'LW_OUT_Consolide': 'R_l_up (W/m²)',
                'PA_Consolide': 'Pa (kPa)'
            }, inplace=True)
            
            df_export.to_csv(csv_out, index=False, float_format="%.2f")
            LOGGER.info(f"   ✅ Succès : {csv_out} enregistré ({len(df_export)} lignes).")
            
        except Exception as e:
            LOGGER.error(f"   ❌ Échec pour {site}: {e}")

if __name__ == "__main__":
    main()
