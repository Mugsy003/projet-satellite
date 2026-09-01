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
TIME_OF_INTEREST = "2025-01-01/2026-01-01"

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
   "Gebesee": {"lon": 10.914411, "lat": 51.100012},
   "Selhausen": {"lon": 6.447118, "lat": 50.865906},
   "Lonzee": {"lon": 4.745863, "lat": 50.551463},
   "Voulundgaard": {"lon": 9.1604, "lat": 56.037431},
   "Klingenberg": { "lon": 13.5223,"lat": 50.893044},
   "Estrees-Mons": {"lon": 3.02065, "lat": 49.87211},
   "Borgo Cioffi": {"lon": 14.957, "lat": 40.523},
   "Grignon": {"lon": 1.952, "lat":  48.844},
   "Lamasquere": {"lon": 1.237878, "lat": 43.496437},
   # --- Stations NOAA SURFRAD ---
   #"Bondville": {"lon": -88.37309, "lat": 40.05192},
   #"Goodwin_Creek": {"lon": -89.8729, "lat": 34.25473},
}

# --- Paramètres TTME (Long & Singh, 2012) ---
# Les seuils NDVI et rugosités (Z0M, Z0H) sont désormais calculés dynamiquement
# par le modèle TTME à partir de chaque scène satellitaire.
# Seuls les paramètres physiques non-dynamiques sont conservés ici.
SITE_TTME_PARAMS = {
    "default": {
        "C_G_SOIL": 0.30,   # Fraction du Rn partant en flux de chaleur sol (sol nu)
        "C_G_VEG": 0.05,    # Fraction du Rn partant en flux de chaleur sol (canopée)
        "EMISSIVITY": 0.95   # Émissivité de surface pour le calcul de LST_Calculee
    },
    "Gebesee": {
        "EMISSIVITY": 0.98
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

PIDS_ICOS_L2_FLUXES = {
    'Lamasquere': 'FeRI6YAnBB_aJ2XTCrDhreXx',
    'Lonzee': 'ZBxsi7rIpGVZqwqrcGQtkrgc',
    'Gebesee': 'NSoD7vTW8nCxCLDfk8qzZjOG',
    'Voulundgaard': '7kPmVcMo_x8jGNWG29pVFAUG',
    'Selhausen': '1WjKCaY068cXzcHKgLbTE-YI',
    'Grignon': 'bEjMUuuYihCIxxm5nS8VrXdw',
    'Borgo Cioffi': 'QF40ymHFpd0aHjhGyKztLMYp',
    'Klingenberg': 'wo3-IvIRLi9cPyDI3d-mOe7z',
    'Estrees-Mons': 'YnZlki8zBvNBivdGpfd1w930',
}

# --- PIDs / Codes des stations NOAA SURFRAD ---
PIDS_NOAA = {
    "Bondville": "bon",
    "Table_Mountain": "tbl",
    "Goodwin_Creek": "gwn"
}

# --- Période de Visualisation (pour visualisation_performances.py) ---
FILTRER_DATES_VISU = False
DATE_DEBUT_VISU = "2022-01-01"
DATE_FIN_VISU = "2024-01-01"

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
    "sharpening_dms_landsat": False,
    "sharpening_tsharp_landsat": False,
    "sharpening_dms_sentinel3": False,
    "fusion_dms": False,
    "fusion_tsharp": False,
    "fusion_dms_s3_s2": False,
    "comparaison_icos": True,
    "visualisation": True,
    "train_lstm_forecast": False,
}

# --- Paramètres Modèle LSTM ---
LSTM_LOOKBACK = 20
LSTM_FORECAST = 7
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.1

# --- Paramètres Entraînement LSTM ---
LSTM_EPOCHS = 30
LSTM_LR = 0.00005
LSTM_WEIGHT_DECAY = 0.0002
