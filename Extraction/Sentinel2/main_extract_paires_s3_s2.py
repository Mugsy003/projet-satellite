import os
import json
from datetime import timedelta
import Extraction.Sentinel2.stac_client_sentinel as stac_s2
import Extraction.Sentinel3.stac_client_sentinel3 as stac_s3
from config import LOGGER, SITES_PILOTES, TIME_OF_INTEREST, OUTPUT_DIR, TIME_MARGIN_MINUTES, ltd, radius_km
from Utils.geo import get_bbox_from_point

def main():
    LOGGER.info("=====================================================")
    LOGGER.info("🚀 DÉMARRAGE EXTRACTION PAIRES (Sentinel-3 + Sentinel-2)")
    LOGGER.info("=====================================================")

    catalog_pc = stac_s3.connect_to_catalog()

    for nom_site, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n🌍 === Recherche de Paires S3/S2 pour : {nom_site} ===")
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km)

        # 1. Chercher S2 (Optique HD)
        items_s2 = stac_s2.search_images_s2(catalog_pc, bbox, TIME_OF_INTEREST, nom_site)
        
        # 2. Chercher S3 SLSTR (Thermique)
        items_s3 = stac_s3.search_images_s3_lst(catalog_pc, bbox, TIME_OF_INTEREST, nom_site)

        if not items_s2 or not items_s3:
            LOGGER.warning(f"   ⚠️ Données insuffisantes pour {nom_site} (S2: {len(items_s2)}, S3: {len(items_s3)})")
            continue

        # 3. Filtrer S2 par nuages
        items_s2_clairs = [i for i in items_s2 if i.properties.get("eo:cloud_cover", 100) <= ltd]
        LOGGER.info(f"   🌤️ S2 sans nuages (<{ltd}%) : {len(items_s2_clairs)} images.")

        # 4. Appariement
        couples = []
        for s2 in items_s2_clairs:
            time_s2 = s2.datetime
            if not time_s2: continue
            
            for s3 in items_s3:
                time_s3 = s3.datetime
                if not time_s3: continue
                
                ecart = abs((time_s2 - time_s3).total_seconds()) / 60.0
                if ecart <= TIME_MARGIN_MINUTES:
                    couples.append({
                        "s2": s2.to_dict(),
                        "s3": s3.to_dict(),
                        "ecart_minutes": ecart,
                        "date": time_s2.strftime("%Y-%m-%d")
                    })
                    break # Passe à l'image S2 suivante

        LOGGER.info(f"   🤝 {len(couples)} paires S3/S2 trouvées (tolérance {TIME_MARGIN_MINUTES} min).")

        if couples:
            chemin_manifeste = os.path.join(OUTPUT_DIR, f"manifest_paires_S3_S2_{nom_site}.json")
            with open(chemin_manifeste, "w") as f:
                json.dump(couples, f, indent=4)
            LOGGER.info(f"   💾 Manifeste paires sauvegardé : {chemin_manifeste}")

    LOGGER.info("\n✅ EXTRACTION DES PAIRES S3/S2 TERMINÉE.")

if __name__ == "__main__":
    main()
