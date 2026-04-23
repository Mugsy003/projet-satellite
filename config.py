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
TIME_OF_INTEREST = "2022-01-01/2022-12-31"
BANDS_OF_INTEREST = ["nir08", "red", "green", "blue", "qa_pixel", "lwir11","swir16"]
TIME_MARGIN_MINUTES = 20
lt = 99
ltd = 30
radius_km = 3
nb_images = 20
max_nuages_rejet = 70
max_jours_fusion = 0
min_couv_rejet = 40
couverture_parfaite = 95

SITES_PILOTES = { 
    "Lamasquere": {"lon": 1.237878, "lat": 43.496437},
    "Lonzee": {"lon":4.745863, "lat": 50.551463},
    "Gebesee": {"lon": 10.914411, "lat": 51.100012},
    "Voulundgaard": {"lon": 9.1604, "lat": 56.037431},
    "Selhausen": {"lon": 6.447118, "lat": 50.865906},
}

#    "Portugal": {"lon": -8.01, "lat": 38.181972},
#    "Spain": {"lon": -5.748678, "lat": 36.444714},
#    "Greece": {"lon": 22.080389, "lat": 38.17075},
#    "Italy": {"lon": 7.67369, "lat": 45.017338},
#    "France": {"lon": 1.7465347, "lat": 43.699843},

PIDS_ICOS = {
    "Lamasquere": 'f_bXbunL87WBKYsV_-NRn_dR',
    "Lonzee": 'vPBrbj9zKYuJlfpC4te4EasK ',  #2023-05-31
    "Gebesee": 'oTVsuExSqsHFTRiBOf1HnKl6',
    "Voulundgaard": 'fEpEBISGMjpKEX4AQNzC3OGY', 
    "Selhausen": 'TJJTkxSqrcJuQ5DUdv5TJio4',
}

