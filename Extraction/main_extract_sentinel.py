"""
Extraction/main_extract_sentinel.py
"""
import os
import json
from config import LOGGER, SITES_PILOTES, TIME_OF_INTEREST, PREVIEWS_DIR, OUTPUT_DIR, ltd, radius_km, nb_images
from Utils import get_bbox_from_point
import Extraction.stac_client_sentinel as stac_client_s2

def main():
    LOGGER.info("🚀 DÉBUT DE LA PHASE 1 SENTINEL-2 : EXTRACTION")

    catalog = stac_client_s2.connect_to_catalog()
    
    manifest_stats = {}      
    manifest_extraction = {} 

    for pays, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"🔍 Traitement du site S2 : {pays}")
        
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km)
        
        # 1. On récupère TOUT pour S2
        all_items = stac_client_s2.search_images_s2(catalog, bbox, TIME_OF_INTEREST, pays)
        
        if not all_items:
            continue

        # 2. Filtrage local
        manifest_stats[pays] = [item.id for item in all_items]
        
        # Filtrage selon la couverture nuageuse (Sentinel-2 utilise souvent eo:cloud_cover ou s2:cloud_shadow_percentage)
        items_propres = [it for it in all_items if it.properties.get("eo:cloud_cover", 100) <= ltd]
        items_propres = items_propres[:nb_images]
        manifest_extraction[pays] = [item.id for item in items_propres]

        LOGGER.info(f"   📊 Stats: {len(all_items)} images | 📥 Extraction: {len(items_propres)} images (seuil {ltd}%)")

        # 3. Preview de la meilleure image
        if items_propres:
            items_propres.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
            stac_client_s2.download_preview_s2(items_propres[0], pays, PREVIEWS_DIR)

    # 4. Sauvegarde
    path_stats = os.path.join(OUTPUT_DIR, "manifest_stats_global_s2.json")
    path_extract = os.path.join(OUTPUT_DIR, "manifest_extraction_s2.json")

    with open(path_stats, "w", encoding="utf-8") as f:
        json.dump(manifest_stats, f, indent=4)
        
    with open(path_extract, "w", encoding="utf-8") as f:
        json.dump(manifest_extraction, f, indent=4)
        
    LOGGER.info(f"\n✅ PHASE 1 SENTINEL-2 TERMINÉE.")
    LOGGER.info(f"   📈 Manifeste GLOBAL S2 : {path_stats}")
    LOGGER.info(f"   ⚙️  Manifeste FILTRÉ S2 : {path_extract}")

if __name__ == "__main__":
    main()
