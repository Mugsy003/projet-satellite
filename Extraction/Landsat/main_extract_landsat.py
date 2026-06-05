"""
Extraction/main_extract.py
"""
import os
import json
from config import LOGGER, SITES_PILOTES, TIME_OF_INTEREST, PREVIEWS_DIR, OUTPUT_DIR, ltd, radius_km, nb_images, ignorer_existants
from Utils import get_bbox_from_point
import Extraction.Landsat.stac_client_landsat as stac_client


def main():
    LOGGER.info("🚀 DÉBUT DE LA PHASE 1 : EXTRACTION (MODE DOUBLE MANIFESTE)")

    catalog = stac_client.connect_to_catalog()
    
    manifest_dates = {}
    if ignorer_existants:
        path_dates = os.path.join(OUTPUT_DIR, "manifest_dates_existantes.json")
        if os.path.exists(path_dates):
            with open(path_dates, "r", encoding="utf-8") as f:
                manifest_dates = json.load(f)
            LOGGER.info("ℹ️ L'option 'ignorer_existants' est activée. Le script ignorera les dates déjà traitées.")
        else:
            LOGGER.warning("⚠️ L'option 'ignorer_existants' est activée, mais aucun fichier 'manifest_dates_existantes.json' n'a été trouvé.")
    
    manifest_stats = {}      # Pour ton script cloud_statistics.py (Toutes les images)
    manifest_extraction = {} # Pour la Phase 2 Transform (Seulement images propres)

    for pays, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"🔍 Traitement du site : {pays}")
        
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km)
        
        # 1. On récupère TOUT (via le nouveau stac_client)
        all_items = stac_client.search_images(catalog, bbox, TIME_OF_INTEREST, pays)
        
        if not all_items:
            continue

        # 2. Filtrage local
        # Liste pour les stats : on prend tout
        manifest_stats[pays] = [item.id for item in all_items]
        
        dates_a_ignorer = manifest_dates.get(pays, [])
        
        # Filtrer selon 'ltd' ET les dates existantes
        items_propres = []
        for it in all_items:
            # Vérifier la couverture nuageuse
            if it.properties.get("eo:cloud_cover", 100) > ltd:
                continue
                
            # Vérifier si on doit ignorer cette date
            if ignorer_existants and dates_a_ignorer and it.datetime:
                date_str = it.datetime.strftime("%Y-%m-%d")
                if date_str in dates_a_ignorer:
                    continue
                    
            items_propres.append(it)
            
        items_propres = items_propres[:nb_images]
        manifest_extraction[pays] = [item.id for item in items_propres]

        LOGGER.info(f"   📊 Stats: {len(all_items)} images | 📥 Extraction: {len(items_propres)} images retenues (seuil {ltd}%" + (f", ignorées: {len(dates_a_ignorer)}" if ignorer_existants else "") + ")")

        # 3. Preview de la meilleure image (la moins nuageuse)
        if items_propres:
            # On trie par couverture nuageuse pour être sûr d'avoir la plus belle preview
            items_propres.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
            stac_client.download_preview(items_propres[0], pays, PREVIEWS_DIR)

    # 4. Sauvegarde des deux manifestes
    path_stats = os.path.join(OUTPUT_DIR, "manifest_stats_global.json")
    path_extract = os.path.join(OUTPUT_DIR, "manifest_extraction.json")

    with open(path_stats, "w", encoding="utf-8") as f:
        json.dump(manifest_stats, f, indent=4)
        
    with open(path_extract, "w", encoding="utf-8") as f:
        json.dump(manifest_extraction, f, indent=4)
        
    LOGGER.info(f"\n✅ PHASE 1 TERMINÉE.")
    LOGGER.info(f"   📈 Manifeste GLOBAL (Stats) : {path_stats}")
    LOGGER.info(f"   ⚙️  Manifeste FILTRÉ (Transform) : {path_extract}")

if __name__ == "__main__":
    main()