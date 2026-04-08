"""
Extraction/main_extract.py
Point d'entrée de la phase d'extraction.
Exécute les recherches STAC, télécharge les previews et génère un Manifeste (JSON)
contenant les IDs des images sélectionnées pour la phase de Transformation.
"""
import os
import json

# Imports depuis la racine du projet
from config import LOGGER, SITES_PILOTES, TIME_OF_INTEREST, PREVIEWS_DIR, OUTPUT_DIR, lt
from Utils import get_bbox_from_point

# Imports locaux du module Extraction
import Extraction.stac_client as stac_client

def main():
    LOGGER.info("🚀 DÉBUT DE LA PHASE 1 : EXTRACTION")

    catalog = stac_client.connect_to_catalog()
    
    # Dictionnaire qui va stocker notre "passage de relais"
    manifest_data = {}

    for pays, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"🔍 Extraction pour le site : {pays}")
        
        # 1. Calcul de la zone (BBox)
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], 3)
        LOGGER.info(f"   BBox = {bbox}")

        # 2. Recherche des images
        mes_items = stac_client.search_images(catalog, bbox, TIME_OF_INTEREST, pays, lt)
        
        if not mes_items:
            LOGGER.warning(f"   ⚠️ Aucune image exploitable trouvée pour {pays}.")
            continue
            
        # 3. Téléchargement de la preview de la meilleure image 
        stac_client.download_preview(mes_items[0], pays, PREVIEWS_DIR)

        # 4. Enregistrement des IDs pour le manifeste
        # On ne stocke que les identifiants (des chaînes de caractères), pas les objets complexes
        liste_ids = [item.id for item in mes_items]
        manifest_data[pays] = liste_ids

    # 5. Sauvegarde du manifeste JSON
    chemin_manifeste = os.path.join(OUTPUT_DIR, "manifest_extraction.json")
    
    with open(chemin_manifeste, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)
        
    LOGGER.info(f"\n✅ PHASE 1 TERMINÉE. Manifeste généré : {chemin_manifeste}")
    LOGGER.info("Prêt pour la phase de Transformation")


if __name__ == "__main__":
    main()