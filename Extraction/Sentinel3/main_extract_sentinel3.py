import os
import json
from datetime import timedelta
from config import LOGGER, SITES_PILOTES, TIME_OF_INTEREST, OUTPUT_DIR, TIME_MARGIN_MINUTES, ltd, radius_km
from Utils.geo import get_bbox_from_point
import Extraction.Sentinel3.stac_client_sentinel3 as stac_s3

def main():
    LOGGER.info("========================================")
    LOGGER.info("🚀 DÉMARRAGE EXTRACTION SENTINEL-3 (Thermique + Optique)")
    LOGGER.info("========================================")
    
    catalog = stac_s3.connect_to_catalog()
    
    for nom_site, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n🌍 === Recherche S3 pour le site : {nom_site} ===")
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km)
        
        # 1. Chercher S3 SLSTR LST (Thermique)
        items_thermique = stac_s3.search_images_s3_lst(catalog, bbox, TIME_OF_INTEREST, nom_site)
        
        # 2. Chercher S3 Synergy (Optique)
        items_optique = stac_s3.search_images_s3_olci(catalog, bbox, TIME_OF_INTEREST, nom_site)
        
        # 3. Appariement basé sur le temps (même satellite, même passage)
        couples_valides = []
        for item_t in items_thermique:
            time_t = item_t.datetime
            if not time_t: continue
            
            for item_o in items_optique:
                time_o = item_o.datetime
                if not time_o: continue
                
                ecart_minutes = abs((time_t - time_o).total_seconds()) / 60.0
                if ecart_minutes <= TIME_MARGIN_MINUTES:
                    # On a un couple complet (Thermique S3 + Optique S3)
                    couples_valides.append({
                        "thermique": item_t.id,
                        "optique": item_o.id,
                        "ecart_minutes": ecart_minutes,
                        "date": time_t.strftime("%Y-%m-%d")
                    })
                    break # Passer au prochain item thermique
        
        LOGGER.info(f"   ✅ {len(couples_valides)} paires complètes S3 (SLSTR + Synergy) trouvées.")
        
        # 4. Sauvegarde du manifeste principal
        if couples_valides:
            chemin_manifeste = os.path.join(OUTPUT_DIR, f"manifest_extraction_S3_{nom_site}.json")
            with open(chemin_manifeste, "w") as f:
                json.dump(couples_valides, f, indent=4)
            LOGGER.info(f"   💾 Manifeste sauvegardé : {chemin_manifeste}")
            
    LOGGER.info("\n✅ EXTRACTION SENTINEL-3 TERMINÉE.")

if __name__ == "__main__":
    main()
