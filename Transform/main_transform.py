"""
Transform/main_transform.py
Point d'entrée de la phase de transformation.
Lit le manifeste JSON, récupère les métadonnées STAC, traite les cubes (ETL)
et génère les cartes et données prêtes pour le Machine Learning.
"""
import os
import json
import pystac_client
import planetary_computer

# Imports depuis la racine
from config import LOGGER, SITES_PILOTES, BANDS_OF_INTEREST, OUTPUT_DIR
from Utils import get_bbox_from_point

# Imports locaux du module
import Transform.processor as processor
import Transform.visualizer as visualizer
def main():
    LOGGER.info("🚀 DÉBUT DE LA PHASE 2 : TRANSFORMATION")
    
    chemin_manifeste = os.path.join(OUTPUT_DIR, "manifest_extraction.json")
    
    # 1. Vérification que la Phase 1 a bien été exécutée
    if not os.path.exists(chemin_manifeste):
        LOGGER.error(f"❌ Impossible de trouver le manifeste : {chemin_manifeste}")
        LOGGER.error("Veuillez exécuter 'python -m Extraction.main_extract' en premier.")
        return

    # 2. Lecture du "passage de relais"
    with open(chemin_manifeste, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # 3. Reconnexion à Microsoft pour récupérer les objets STAC à partir de leurs IDs
    LOGGER.info("🌐 Reconnexion au catalogue STAC pour reconstituer les items...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    mes_images_arrays = {}

    # 4. Boucle de traitement par pays
    for pays, liste_ids in manifest_data.items():
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"⚙️ Transformation pour le site : {pays}")
        
        if not liste_ids:
            LOGGER.warning(f"   ⚠️ Aucun ID trouvé dans le manifeste pour {pays}.")
            continue
            
        # A. Recalculer la BBox pour ce pays à partir de ses coordonnées
        coords = SITES_PILOTES[pays]
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km=3)
        
        # B. Retrouver les objets STAC exacts demandés par la Phase 1
        search = catalog.search(collections=["landsat-c2-l2"], ids=liste_ids)
        mes_items = list(search.items())
        
        # C. Lancer la machinerie lourde (Mode Time-Series)
        liste_images = processor.process_satellite_timeseries(mes_items, bbox, BANDS_OF_INTEREST)
        
        if not liste_images:
            continue
            
        # D. & E. Sauvegarder la série complète
        visualizer.save_timeseries_results(liste_images, pays, OUTPUT_DIR)
        visualizer.save_indices_maps(liste_images, pays, OUTPUT_DIR)
        # Pour les histogrammes comparatifs de la fin, on peut décider de stocker 
        # uniquement la TOUTE PREMIÈRE image valide de la série temporelle
        mes_images_arrays[pays] = liste_images[0]["reflectance"]

    # 5. Étape finale : Génération des courbes comparatives et des cartes d'indices pour TOUTES les séries temporelles
    visualizer.generate_comparative_histograms(mes_images_arrays, OUTPUT_DIR)
    

    LOGGER.info("\n✅ PHASE 2 TERMINÉE. Données transformées avec succès !")
    LOGGER.info("Prêt pour la phase d'entraînement")

if __name__ == "__main__":
    main()