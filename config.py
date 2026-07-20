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
TIME_OF_INTEREST = "2022-01-01/2023-01-01"

BANDS_OF_INTEREST = ["nir08", "red", "green", "blue", "qa_pixel", "lwir11","swir16", "swir22"]
BANDS_OF_INTEREST_S2 = ["B02", "B03", "B04", "B08", "B11", "SCL"]
TIME_MARGIN_MINUTES = 60
lt = 99
ltd = 99  # Seuil augmenté pour ne pas rater d'images claires localement
radius_km = 3
radius_km_s3 = 25  # Rayon élargi pour S3 
nb_images = 500
max_nuages_rejet = 70
max_jours_fusion = 0
min_couv_rejet = 30
couverture_parfaite = 95

# Si True, l'extraction Landsat ignorera les images dont les dates sont déjà 
# listées dans Outputs/manifest_dates_existantes.json pour aller en chercher de nouvelles.
ignorer_existants = False

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

# --- Paramètres TTME spécifiques par site ---
SITE_TTME_PARAMS = {
    "default": {
        "NDVI_SOL": 0.15,
        "NDVI_VEG": 0.90,
        "Z0M_SOIL": 0.005,
        "Z0H_SOIL": 0.0005,
        "Z0M_VEG": 0.10,
        "Z0H_VEG": 0.01,
        "C_G_SOIL": 0.30,
        "C_G_VEG": 0.05,
        "EMISSIVITY": 0.95  # Ajusté de 0.98 à 0.95 pour réchauffer LST_Calculee
    },
    "Grignon": {
        "NDVI_SOL": 0.15,
        "NDVI_VEG": 0.85,
        "Z0M_SOIL": 0.01,
        "Z0H_SOIL": 0.001,
        "Z0M_VEG": 0.15,   
        "Z0H_VEG": 0.015,
        "C_G_SOIL": 0.30,
        "C_G_VEG": 0.05,
        "EMISSIVITY": 0.96
    },
    "Borgo Cioffi": {
        "NDVI_SOL": 0.12,
        "NDVI_VEG": 0.88,
        "Z0M_SOIL": 0.005,
        "Z0H_SOIL": 0.0005,
        "Z0M_VEG": 0.12,
        "Z0H_VEG": 0.012,
        "C_G_SOIL": 0.35,  
        "C_G_VEG": 0.05,
        "EMISSIVITY": 0.94  # Sol plus sec, émissivité plus faible
    }
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

# --- Période de Visualisation (pour visualisation_performances.py) ---
FILTRER_DATES_VISU = True
DATE_DEBUT_VISU = "2022-01-01"
DATE_FIN_VISU = "2023-01-01"

# --- Pipeline : étapes à exécuter via main.py ---
PIPELINE_STEPS = {
    "extraction_landsat": True,
    "extraction_ecostress": False,
    "extraction_sentinel": False,
    "extraction_sentinel3": False,
    "extraction_paires_eco_s2": False,
    "extraction_paires_s3_s2": False,
    "transform_landsat": True,
    "transform_ecostress": False,
    "transform_sentinel": False,
    "transform_sentinel3": False,
    "sharpening_dms_landsat": True,
    "sharpening_tsharp_landsat": False,
    "sharpening_dms_sentinel3": False,
    "fusion_dms": False,
    "fusion_tsharp": False,
    "fusion_dms_s3_s2": False,
    "comparaison_icos": True,
    "visualisation": True,
}
