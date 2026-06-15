"""
Transform/Fusion/main_transform_paires_s3_s2.py
===============================================
Orchestrateur qui traite les paires croisées S3/S2 trouvées par l'extraction.
Il télécharge/transforme à la fois la LST Sentinel-3 (1km) et les indices Sentinel-2 (10m)
pour chaque paire valide, préparant le terrain pour la fusion DMS.
"""
import os
import json
import pystac_client
import planetary_computer
import rasterio

from config import LOGGER, SITES_PILOTES, OUTPUT_DIR, radius_km_s3, BANDS_OF_INTEREST_S2
from Utils.geo import get_bbox_from_point

import Transform.Sentinel3.processor_sentinel3 as processor_s3
import Transform.Sentinel2.processor_sentinel as processor_s2
import Transform.common.visualizer as visualizer

def main():
    LOGGER.info("========================================")
    LOGGER.info("🚀 DÉMARRAGE TRANSFORMATION PAIRES S3+S2")
    LOGGER.info("========================================")

    # Reconnexion à Planetary Computer pour S2 (car le json contient des ID STAC ou les items bruts)
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    for nom_site, coords in SITES_PILOTES.items():
        chemin_manifeste = os.path.join(OUTPUT_DIR, f"manifest_paires_S3_S2_{nom_site}.json")
        if not os.path.exists(chemin_manifeste):
            continue

        LOGGER.info(f"\n🌍 === Traitement des paires pour : {nom_site} ===")
        
        # Bbox pour S3 (large)
        bbox_s3 = get_bbox_from_point(coords["lon"], coords["lat"], radius_km_s3)
        # Bbox pour S2 (restreinte, comme dans config globale)
        bbox_s2 = get_bbox_from_point(coords["lon"], coords["lat"], radius_km=3)

        dossier_s3 = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{nom_site}_S3", "TIF_Data")
        os.makedirs(dossier_s3, exist_ok=True)

        with open(chemin_manifeste, "r", encoding="utf-8") as f:
            paires = json.load(f)

        LOGGER.info(f"   📋 {len(paires)} paires à traiter.")
        
        # On traite toutes les paires disponibles

        for i, paire in enumerate(paires):
            date_str = paire["date"]
            prefixe_s3 = f"{date_str}_{nom_site}"
            LOGGER.info(f"\n   📅 [{i+1}/{len(paires)}] Paire du {date_str} (Ecart {paire['ecart_minutes']:.0f} min)")

            # 1. Traitement S3 (Thermique uniquement nécessaire pour la fusion)
            LOGGER.info("      -> Traitement Sentinel-3 (LST 1km)...")
            
            # Récupération de l'ID S3
            item_s3 = paire["s3"]
            s3_id = item_s3 if isinstance(item_s3, str) else item_s3.get("id")
            
            # Requête STAC pour avoir un token frais
            search_s3 = catalog.search(collections=["sentinel-3-slstr-lst-l2-netcdf"], ids=[s3_id])
            mes_items_s3 = list(search_s3.items())
            if not mes_items_s3:
                LOGGER.warning(f"      ⚠️ Impossible de récupérer l'item S3 {s3_id} (peut-être hors ligne)")
                continue
                
            s3_frais = planetary_computer.sign(mes_items_s3[0].to_dict())
            
            tif_lst = processor_s3.process_slstr_lst(s3_frais, dossier_s3, prefixe_s3, bbox_s3)
            
            if tif_lst is None:
                # Échec S3 (souvent bbox hors zone valide), on passe S2
                continue
            
            # 2. Traitement S2 (Optique)
            LOGGER.info("      -> Traitement Sentinel-2 (Optique 10m)...")
            
            import glob
            dossier_s2 = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")
            pattern_s2 = os.path.join(dossier_s2, f"{date_str[:10]}*_{nom_site}*_NDVI.tif")
            if glob.glob(pattern_s2):
                LOGGER.info("      ✅ Indices Sentinel-2 déjà présents pour cette date. Skip.")
                continue

            # Le manifeste contient soit l'item dict entier, soit son ID.
            item_s2 = paire["s2"]
            if isinstance(item_s2, str):
                s2_id = item_s2
            else:
                s2_id = item_s2.get("id")

            # Récupérer l'item frais depuis STAC pour avoir des liens signés valides
            search = catalog.search(collections=["sentinel-2-l2a"], ids=[s2_id])
            mes_items_s2 = search.item_collection()
            
            if not mes_items_s2:
                LOGGER.warning(f"      ⚠️ Impossible de récupérer l'item S2 {s2_id}")
                continue

            planetary_computer.sign_inplace(mes_items_s2)

            # Utiliser le processeur S2
            # Attention: processeur S2 écrit dans un chemin hardcodé dans le visualizer,
            # mais on peut forcer son exécution sans filtrage fort de nuages pour ce test croisé
            liste_images = processor_s2.process_s2_timeseries(
                mes_items=mes_items_s2, 
                bbox=bbox_s2, 
                bands_of_interest=BANDS_OF_INTEREST_S2, 
                max_jours_fusion=0,
                max_nuages_rejet=100,  # on accepte les nuages ici pour garantir que le test tourne
                min_couv_rejet=0,
                couverture_parfaite=100
            )

            if liste_images:
                # Sauvegarde avec les noms conformes S2
                visualizer.save_indices_maps(liste_images, nom_site + "_S2", OUTPUT_DIR)
                LOGGER.info("      ✅ Transformation S2 réussie.")
            else:
                LOGGER.error("      ❌ Échec de la transformation S2.")

    LOGGER.info("\n✅ TRANSFORMATION PAIRES S3+S2 TERMINÉE.")


if __name__ == "__main__":
    with rasterio.Env(GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=5, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".TIF"):
        main()
