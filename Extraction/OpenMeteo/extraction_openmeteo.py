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
API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

def fetch_openmeteo_forecast(site_name, lat, lon):
    """
    Récupère les prévisions météorologiques historiques pour un site donné.
    """
    LOGGER.info(f"   📥 Extraction Open-Meteo pour {site_name} ({lat}, {lon})...")
    
    # Requête de l'API
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "surface_pressure", "shortwave_radiation"],
        "timezone": "UTC"
    }
    
    response = requests.get(API_URL, params=params)
    
    if response.status_code != 200:
        LOGGER.error(f"❌ Erreur de l'API Open-Meteo pour {site_name}: {response.text}")
        return None
        
    data = response.json()
    
    if "hourly" not in data:
        LOGGER.error(f"❌ Pas de données horaires trouvées pour {site_name}")
        return None
        
    hourly = data["hourly"]
    
    # Création du DataFrame
    df = pd.DataFrame({
        "TIMESTAMP": pd.to_datetime(hourly["time"]),
        "Ta (°C)": hourly["temperature_2m"],
        "RH (%)": hourly["relative_humidity_2m"],
        "u (m/s)": hourly["wind_speed_10m"],
        "surface_pressure_hPa": hourly["surface_pressure"],
        "R_s_down (W/m²)": hourly["shortwave_radiation"]
    })
    
    # Conversion de pression (hPa vers kPa pour correspondre à ERA5)
    df["Pa (kPa)"] = df["surface_pressure_hPa"] / 10.0
    df = df.drop(columns=["surface_pressure_hPa"])
    
    # Retirer les jours incomplets ou futurs s'il n'y a pas de prévisions
    df = df.dropna()
    
    return df

def main():
    LOGGER.info("🚀 Démarrage de l'extraction Open-Meteo Historical Forecast")
    
    out_folder = os.path.join(OUTPUT_DIR, "Extraction", "OpenMeteo")
    os.makedirs(out_folder, exist_ok=True)
    
    for site, coords in SITES_PILOTES.items():
        df = fetch_openmeteo_forecast(site, coords["lat"], coords["lon"])
        if df is not None and not df.empty:
            out_file = os.path.join(out_folder, f"donnees_openmeteo_{site}.csv")
            df.to_csv(out_file, index=False)
            LOGGER.info(f"      ✅ Données sauvegardées dans {out_file} ({len(df)} lignes)")
            
    LOGGER.info("🎯 Extraction terminée.")

if __name__ == "__main__":
    main()
