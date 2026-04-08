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
TIME_OF_INTEREST = "2025-06-15/2025-09-15"
BANDS_OF_INTEREST = ["nir08", "red", "green", "blue", "qa_pixel", "lwir11","swir16"]
lt = 20

SITES_PILOTES = { 
    "Portugal": {"lon": -8.01, "lat": 38.181972},
    "Spain": {"lon": -5.748678, "lat": 36.444714},
    "Greece": {"lon": 22.080389, "lat": 38.17075},
    "Italy": {"lon": 7.67369, "lat": 45.017338},
    "France": {"lon": 1.7465347, "lat": 43.699843}
}