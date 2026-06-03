"""
config.py
Configuration globale du projet. Centralise les constantes, 
les chemins de fichiers et le paramétrage du logger.
"""
import os
import logging

# --- Configuration du Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
LOGGER = logging.getLogger(__name__)

# --- Chemins et Dossiers ---
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "Outputs")
PREVIEWS_DIR = os.path.join(BASE_DIR, "Previews_Landsat")

# Création automatique des dossiers
for directory in [OUTPUT_DIR, PREVIEWS_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Constantes du Projet ---
TIME_OF_INTEREST = "2022-12-31/2024-01-01"

BANDS_OF_INTEREST = ["nir08", "red", "green", "blue", "qa_pixel", "lwir11","swir16"]
BANDS_OF_INTEREST_S2 = ["B02", "B03", "B04", "B08", "B11", "SCL"]
TIME_MARGIN_MINUTES = 30
lt = 99
ltd = 30
radius_km = 3
nb_images = 30
max_nuages_rejet = 70
max_jours_fusion = 0
min_couv_rejet = 40
couverture_parfaite = 95

# Si True, l'extraction Landsat ignorera les images dont les dates sont déjà 
# listées dans Outputs/manifest_dates_existantes.json pour aller en chercher de nouvelles.
ignorer_existants = True

SITES_PILOTES = { 
   "Greece": {"lon": 22.080389, "lat": 38.17075},
   "Gebesee": {"lon": 10.914411, "lat": 51.100012},
   "Selhausen": {"lon": 6.447118, "lat": 50.865906},
   "Italy": {"lon": 7.67369, "lat": 45.017338},
   "Lonzee": {"lon": 4.745863, "lat": 50.551463},
   "Voulundgaard": {"lon": 9.1604, "lat": 56.037431},
   "Klingenberg": { "lon": 13.5223,"lat": 50.893044},
   "Estrees-Mons": {"lon": 3.02065, "lat": 49.87211},
   "Borgo Cioffi": {"lon": 14.957, "lat": 40.523},
   "Grignon": {"lon": 1.952, "lat":  48.844},
   "Lamasquere": {"lon": 1.237878, "lat": 43.496437},
   # --- Stations NOAA SURFRAD ---
   "Bondville": {"lon": -88.37309, "lat": 40.05192},
   "Goodwin_Creek": {"lon": -89.8729, "lat": 34.25473},
   
}



PIDS_ICOS = {
    "Lamasquere": 'tZlz-zEjgsdC11OtOL2Ijz6Z',
    "Lonzee": 'vPBrbj9zKYuJlfpC4te4EasK',
    "Gebesee": 'oTVsuExSqsHFTRiBOf1HnKl6',
    "Voulundgaard": 'fEpEBISGMjpKEX4AQNzC3OGY', 
    "Selhausen": 'TJJTkxSqrcJuQ5DUdv5TJio4',
    "Grignon": "WXvfWDja4xMP9n_rF8fl0Xxp",
    "Borgo Cioffi": "6fkal4WFXkNLxlrGXRmHrif_",
    "Klingenberg": "WJpiQa9U59h5v204K77o4wJY",
    "Estrees-Mons": "Fowem-0Vzv3g3zElIav5uxO8"
}

# --- PIDs / Codes des stations NOAA SURFRAD ---
PIDS_NOAA = {
    "Bondville": "bon",
    "Table_Mountain": "tbl",
    "Goodwin_Creek": "gwn"
}


