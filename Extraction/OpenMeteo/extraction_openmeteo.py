import os
import sys
import logging
import requests
import pandas as pd
import numpy as np

# Adjust path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import OUTPUT_DIR, SITES_PILOTES

# --- Configuration du Logger ---
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "extraction_openmeteo.log"), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
LOGGER = logging.getLogger(__name__)

# --- Constantes ---
API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

def generate_grid(lat, lon, offset=0.1):
    """Génère une grille 2x2 autour du point central pour faire une moyenne spatiale."""
    lats = [lat - offset, lat + offset]
    lons = [lon - offset, lon + offset]
    
    grid_lats = []
    grid_lons = []
    for la in lats:
        for lo in lons:
            grid_lats.append(round(la, 4))
            grid_lons.append(round(lo, 4))
            
    return grid_lats, grid_lons

def fetch_openmeteo_forecast(site_name, center_lat, center_lon):
    """
    Récupère les VRAIES prévisions météorologiques (Previous Runs) pour un site,
    en moyennant sur une grille de 4 pixels (2x2).
    """
    LOGGER.info(f"   📥 Extraction Open-Meteo Previous Runs pour {site_name} (Grille 2x2)...")
    
    lats, lons = generate_grid(center_lat, center_lon)
    
    # Construction des variables
    hourly_vars = []
    for d in range(1, 8):
        hourly_vars.append(f"temperature_2m_previous_day{d}")
        hourly_vars.append(f"relative_humidity_2m_previous_day{d}")
        hourly_vars.append(f"shortwave_radiation_previous_day{d}")
        
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": hourly_vars,
        "timezone": "UTC"
    }
    
    import time
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            if response.status_code == 200:
                break
            
            LOGGER.error(f"❌ Erreur {response.status_code} de l'API Open-Meteo pour {site_name}: {response.text}")
            if attempt == max_retries - 1:
                return None
                
            if response.status_code == 429:
                LOGGER.info("⏳ Limite de requêtes atteinte (429). Pause de 65 secondes...")
                time.sleep(65)
            else:
                time.sleep(10)
                
        except Exception as e:
            LOGGER.warning(f"⚠️ Erreur réseau pour {site_name} (tentative {attempt+1}/{max_retries}) : {e}")
            if attempt == max_retries - 1:
                LOGGER.error(f"❌ Échec définitif pour {site_name}.")
                return None
            time.sleep(10)
    
    data = response.json()
    if not isinstance(data, list):
        # S'il n'y a qu'un point, l'API renvoie parfois un dict au lieu d'une liste
        data = [data]
        
    all_dfs = []
    for point_data in data:
        if "hourly" not in point_data:
            continue
            
        hourly = point_data["hourly"]
        df_point = pd.DataFrame({"TIMESTAMP": pd.to_datetime(hourly["time"])})
        
        for d in range(1, 8):
            df_point[f"Ta_fcst_J{d}"] = hourly[f"temperature_2m_previous_day{d}"]
            df_point[f"RH_fcst_J{d}"] = hourly[f"relative_humidity_2m_previous_day{d}"]
            df_point[f"Rs_fcst_J{d}"] = hourly[f"shortwave_radiation_previous_day{d}"]
            
        all_dfs.append(df_point)
        
    if not all_dfs:
        LOGGER.error(f"❌ Aucune donnée valide trouvée pour {site_name}")
        return None
        
    # Concaténer puis faire la moyenne par timestamp
    df_concat = pd.concat(all_dfs)
    df_mean = df_concat.groupby("TIMESTAMP").mean().reset_index()
    
    # Retirer les jours sans prévisions
    df_mean = df_mean.dropna()
    
    return df_mean

def main():
    LOGGER.info("🚀 Démarrage de l'extraction Open-Meteo Previous Runs (Grille Moyennée)")
    
    out_folder = os.path.join(OUTPUT_DIR, "Extraction", "OpenMeteo")
    os.makedirs(out_folder, exist_ok=True)
    
    for site, coords in SITES_PILOTES.items():
        df = fetch_openmeteo_forecast(site, coords["lat"], coords["lon"])
        if df is not None and not df.empty:
            out_file = os.path.join(out_folder, f"donnees_openmeteo_{site}.csv")
            df.to_csv(out_file, index=False)
            LOGGER.info(f"      ✅ Données sauvegardées dans {out_file} ({len(df)} lignes moyennées)")
            
    LOGGER.info("🎯 Extraction terminée.")

if __name__ == "__main__":
    main()
