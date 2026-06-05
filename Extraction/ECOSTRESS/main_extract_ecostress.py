"""
Extraction/main_extract_ecostress.py
"""
import os
import json
from config import LOGGER, SITES_PILOTES, TIME_OF_INTEREST, PREVIEWS_DIR, OUTPUT_DIR, ltd, radius_km, nb_images
from Utils import get_bbox_from_point
import Extraction.ECOSTRESS.stac_client_ecostress as stac_client_eco

# ECOSTRESS a une fauchée étroite (~400 km). Beaucoup de tuiles STAC 
# couvrent géographiquement le site mais n'ont aucune donnée réelle (NaN).
# On prend donc beaucoup plus d'images que pour Landsat/Sentinel.
NB_IMAGES_ECOSTRESS = nb_images * 5  # 100 par défaut

def main():
    LOGGER.info("🚀 DÉBUT DE LA PHASE 1 ECOSTRESS : EXTRACTION")

    catalog = stac_client_eco.connect_to_catalog()
    
    manifest_stats = {}      
    manifest_extraction = {} 

    for pays, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"🔍 Traitement du site ECOSTRESS : {pays}")
        
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km)
        
        # 1. On récupère TOUT pour ECOSTRESS
        all_items = stac_client_eco.search_images_ecostress(catalog, bbox, TIME_OF_INTEREST, pays)
        
        if not all_items:
            continue

        # 2. Filtrage local
        manifest_stats[pays] = [item.id for item in all_items]
        
        # Les métadonnées STAC de NASA CMR pour ECOSTRESS ne contiennent généralement pas eo:cloud_cover.
        # On va donc prendre un nombre fixe d'images ou les accepter toutes jusqu'à nb_images.
        # Pour rester cohérent, on simule un filtre en prenant simplement les nb_images premières.
        items_propres = all_items[:NB_IMAGES_ECOSTRESS]
        manifest_extraction[pays] = [item.id for item in items_propres]

        LOGGER.info(f"   📊 Stats: {len(all_items)} images | 📥 Extraction: {len(items_propres)} images (Pas de filtre nuage en métadonnées)")

        # 3. Preview de la meilleure image (ici la première disponible)
        if items_propres:
            stac_client_eco.download_preview_ecostress(items_propres[0], pays, PREVIEWS_DIR)

    # 4. Sauvegarde
    path_stats = os.path.join(OUTPUT_DIR, "manifest_stats_global_ecostress.json")
    path_extract = os.path.join(OUTPUT_DIR, "manifest_extraction_ecostress.json")

    with open(path_stats, "w", encoding="utf-8") as f:
        json.dump(manifest_stats, f, indent=4)
        
    with open(path_extract, "w", encoding="utf-8") as f:
        json.dump(manifest_extraction, f, indent=4)
        
    LOGGER.info(f"\n✅ PHASE 1 ECOSTRESS TERMINÉE.")
    LOGGER.info(f"   📈 Manifeste GLOBAL ECOSTRESS : {path_stats}")
    LOGGER.info(f"   ⚙️  Manifeste FILTRÉ ECOSTRESS : {path_extract}")

if __name__ == "__main__":
    main()
