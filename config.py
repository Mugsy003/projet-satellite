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
TIME_OF_INTEREST = "2025-01-01/2026-03-01"

BANDS_OF_INTEREST = ["nir08", "red", "green", "blue", "qa_pixel", "lwir11","swir16"]
BANDS_OF_INTEREST_S2 = ["B02", "B03", "B04", "B08", "B11", "SCL"]
TIME_MARGIN_MINUTES = 30
lt = 99
ltd = 30
radius_km = 3
nb_images = 20
max_nuages_rejet = 70
max_jours_fusion = 0
min_couv_rejet = 40
couverture_parfaite = 95

SITES_PILOTES = { 
   "Greece": {"lon": 22.080389, "lat": 38.17075},
   "Gebesee": {"lon": 10.914411, "lat": 51.100012},
   "Selhausen": {"lon": 6.447118, "lat": 50.865906},
   "Italy": {"lon": 7.67369, "lat": 45.017338},
   "Lonzee": {"lon": 4.745863, "lat": 50.551463},
}




PIDS_ICOS = {
    "Lamasquere": 'wAacHZyZSqZyntBZDMZJ5wf3',
    "Lonzee": 'vPBrbj9zKYuJlfpC4te4EasK',
    "Gebesee": 'oTVsuExSqsHFTRiBOf1HnKl6',
    "Voulundgaard": 'fEpEBISGMjpKEX4AQNzC3OGY', 
    "Selhausen": 'TJJTkxSqrcJuQ5DUdv5TJio4',
}

