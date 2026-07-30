import os
import json
import logging
import pystac_client
import planetary_computer
import rasterio

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import LOGGER, SITES_PILOTES, BANDS_OF_INTEREST_S2, OUTPUT_DIR, TIME_OF_INTEREST
from Utils import get_bbox_from_point

import Transform.Sentinel2.processor_sentinel as processor_s2
import Transform.common.visualizer as visualizer

def run_extraction_all_s2():
    LOGGER.info("🚀 DÉBUT DE L'EXTRACTION COMPLÈTE SENTINEL-2 (Indépendante de Landsat)")
    
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    for pays in SITES_PILOTES.keys():
        coords = SITES_PILOTES[pays]
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km=3)
        
        time_of_interest = TIME_OF_INTEREST
        
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"Recherche S2 pour {pays} sur la période {time_of_interest}...")
        
        # On limite la recherche globale aux tuiles avec moins de 80% de nuages pour ne pas récupérer des listes immenses de tuiles inutiles.
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_of_interest,
            query={"eo:cloud_cover": {"lt": 80}}
        )
        
        mes_items = search.item_collection()
        LOGGER.info(f"   🛰️ {len(mes_items)} images trouvées au total sur le catalogue STAC.")
        
        if not mes_items:
            LOGGER.info(f"Aucune image trouvée pour {pays}.")
            continue
            
        planetary_computer.sign_inplace(mes_items)
        
        # max_nuages_rejet=100 (pour désactiver le rejet basé sur eo:cloud_cover global)
        # min_couv_rejet=80 (pour garantir que sur notre bbox, l'image est à 80% CLAIRE, donc max 20% de nuages)
        # max_jours_fusion=0 (pas de réparation, on veut juste l'image brute claire)
        
        liste_images = processor_s2.process_s2_timeseries(
            mes_items=mes_items, 
            bbox=bbox, 
            bands_of_interest=BANDS_OF_INTEREST_S2, 
            max_jours_fusion=0,
            max_nuages_rejet=100, 
            min_couv_rejet=80,
            couverture_parfaite=80  # On accepte tout ce qui est >= 80% sans essayer de fusionner
        )
        
        if liste_images:
            LOGGER.info(f"   💾 Sauvegarde de {len(liste_images)} images pour {pays}_S2...")
            visualizer.save_timeseries_results(liste_images, pays + "_S2", OUTPUT_DIR)
            visualizer.save_indices_maps(liste_images, pays + "_S2", OUTPUT_DIR)
        else:
            LOGGER.info(f"Aucune image ne satisfaisait le critère de couverture pour {pays}.")
    
    LOGGER.info("\n✅ EXTRACTION COMPLÈTE TERMINÉE POUR TOUS LES SITES.")

if __name__ == "__main__":
    with rasterio.Env(
        GDAL_HTTP_MAX_RETRY=10,       
        GDAL_HTTP_RETRY_DELAY=30,      
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF",
        GDAL_HTTP_TIMEOUT=45          
    ):
        run_extraction_all_s2()
