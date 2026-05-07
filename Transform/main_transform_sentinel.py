"""
Transform/main_transform_sentinel.py
"""
import os
import json
import pystac_client
import planetary_computer
import rasterio

from config import LOGGER, SITES_PILOTES, BANDS_OF_INTEREST_S2, OUTPUT_DIR, max_nuages_rejet, min_couv_rejet, couverture_parfaite, max_jours_fusion
from Utils import get_bbox_from_point

import Transform.processor_sentinel as processor_s2
import Transform.visualizer as visualizer

def main():
    LOGGER.info("🚀 DÉBUT DE LA PHASE 2 SENTINEL-2 : TRANSFORMATION")
    
    chemin_manifeste = os.path.join(OUTPUT_DIR, "manifest_extraction_s2.json")
    
    if not os.path.exists(chemin_manifeste):
        LOGGER.error(f"❌ Impossible de trouver le manifeste : {chemin_manifeste}")
        LOGGER.error("Veuillez exécuter 'python -m Extraction.main_extract_sentinel' en premier.")
        return

    with open(chemin_manifeste, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    LOGGER.info("🌐 Reconnexion au catalogue STAC pour reconstituer les items S2...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    mes_images_arrays = {}

    for pays, liste_ids in manifest_data.items():
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"⚙️ Transformation S2 pour le site : {pays}")
        
        if not liste_ids:
            LOGGER.warning(f"   ⚠️ Aucun ID trouvé dans le manifeste pour {pays}.")
            continue
            
        coords = SITES_PILOTES[pays]
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km=3)
        
        import time
        from pystac_client.exceptions import APIError
        search = catalog.search(collections=["sentinel-2-l2a"], ids=liste_ids)
        
        max_retries = 5
        mes_items = None
        for attempt in range(max_retries):
            try:
                mes_items = search.item_collection()
                break
            except APIError as e:
                LOGGER.warning(f"   ⚠️ Erreur API STAC: {e}. Tentative {attempt + 1}/{max_retries}...")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    LOGGER.error(f"   ❌ Échec de la recherche STAC après {max_retries} tentatives.")
            except Exception as e:
                LOGGER.error(f"   ❌ Erreur inattendue lors de la recherche: {e}")
                break
                
        if mes_items is None:
            continue
        
        planetary_computer.sign_inplace(mes_items)
        
        liste_images = processor_s2.process_s2_timeseries(
            mes_items=mes_items, 
            bbox=bbox, 
            bands_of_interest=BANDS_OF_INTEREST_S2, 
            max_jours_fusion=max_jours_fusion,
            max_nuages_rejet=max_nuages_rejet, 
            min_couv_rejet=min_couv_rejet,
            couverture_parfaite=couverture_parfaite
        )
        
        if not liste_images:
            continue
            
        visualizer.save_timeseries_results(liste_images, pays + "_S2", OUTPUT_DIR)
        visualizer.save_indices_maps(liste_images, pays + "_S2", OUTPUT_DIR)
        
        mes_images_arrays[pays] = liste_images[0]["reflectance"]

    # Génération des histogrammes 
    visualizer.generate_comparative_histograms(mes_images_arrays, OUTPUT_DIR)
    
    LOGGER.info("\n✅ PHASE 2 SENTINEL-2 TERMINÉE. Données transformées avec succès !")

def run():
    with rasterio.Env(
        GDAL_HTTP_MAX_RETRY=10,       
        GDAL_HTTP_RETRY_DELAY=30,      
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF",
        GDAL_HTTP_TIMEOUT=45          
    ):
        main()

if __name__ == "__main__":
    run()
