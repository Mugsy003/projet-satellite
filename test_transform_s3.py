"""
Test complet : Transformation S3 (1 paire) + DMS Sharpening S3 (1km -> 300m)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json, os
from config import LOGGER, SITES_PILOTES, OUTPUT_DIR, radius_km_s3
from Utils.geo import get_bbox_from_point
import Transform.Sentinel3.processor_sentinel3 as processor_s3
from Transform.Sentinel3.dms_sharpening_sentinel3 import process_dms_s3

NOM_SITE = "Lamasquere"
coords = SITES_PILOTES[NOM_SITE]
bbox_lonlat = get_bbox_from_point(coords["lat"], coords["lon"], radius_km_s3)

LOGGER.info(f"Rayon S3 : {radius_km_s3} km -> bbox : {bbox_lonlat}")

manifeste = os.path.join(OUTPUT_DIR, f"manifest_extraction_S3_{NOM_SITE}.json")
with open(manifeste, "r") as f:
    paires = json.load(f)

# On teste 1 seule paire
paire = paires[0]
date_str = paire["date"]
prefixe = f"{date_str}_{NOM_SITE}"
dossier_sortie = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{NOM_SITE}_S3", "TIF_Data")
os.makedirs(dossier_sortie, exist_ok=True)

LOGGER.info(f"\n=== ETAPE 1 : Transformation (date {date_str}) ===")
tif_lst = processor_s3.process_slstr_lst(paire["thermique"], dossier_sortie, prefixe, bbox_lonlat)
tif_opt = processor_s3.process_synergy_optique(paire["optique"], dossier_sortie, prefixe, bbox_lonlat)

if tif_lst and tif_opt:
    LOGGER.info(f"\n=== ETAPE 2 : DMS Sharpening S3 (1km -> 300m) ===")
    process_dms_s3(NOM_SITE, date_str, dossier_sortie)
else:
    LOGGER.error("Transformation echouee, impossible de lancer le DMS.")

LOGGER.info("\nTest termine!")
