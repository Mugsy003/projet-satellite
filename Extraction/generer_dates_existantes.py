"""
generer_dates_existantes.py

Script utilitaire qui parcourt les dossiers d'outputs (TIF_Data) pour extraire 
les dates (YYYY-MM-DD) des images satellitaires déjà téléchargées et traitées.
Génère un fichier 'Outputs/manifest_dates_existantes.json' qui peut ensuite
être utilisé par main_extract.py pour ignorer ces dates et aller chercher de 
nouvelles images.
"""

import os
import glob
import re
import json
from config import SITES_PILOTES, OUTPUT_DIR

def main():
    print("Analyse des images deja presentes sur le disque...")
    manifest_dates = {}
    total_dates = 0

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for site in SITES_PILOTES.keys():
        dates_site = set()
        
        # On cherche dans le dossier 3_Indices/TIF_Data (les images transformées)
        # Mais on pourrait aussi chercher dans 1_Raw
        search_pattern = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data", "*.tif")
        files = glob.glob(search_pattern)
        
        for f in files:
            basename = os.path.basename(f)
            # Les fichiers sont au format YYYY-MM-DD_...
            match = re.search(r'^(\d{4}-\d{2}-\d{2})', basename)
            if match:
                dates_site.add(match.group(1))
                
        # Si le dossier 3_Indices est vide ou purgé, on cherche aussi dans 1_Raw au cas où
        search_pattern_raw = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}", "1_Raw", "TIF_Data", "*.tif")
        files_raw = glob.glob(search_pattern_raw)
        
        for f in files_raw:
            basename = os.path.basename(f)
            match = re.search(r'^(\d{4}-\d{2}-\d{2})', basename)
            if match:
                dates_site.add(match.group(1))

        liste_dates = sorted(list(dates_site))
        manifest_dates[site] = liste_dates
        total_dates += len(liste_dates)
        
        print(f"  - {site} : {len(liste_dates)} dates uniques trouvées.")

    output_path = os.path.join(OUTPUT_DIR, "manifest_dates_existantes.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest_dates, f, indent=4)
        
    print(f"\nTermine ! {total_dates} dates au total sauvegardees dans : {output_path}")

if __name__ == "__main__":
    main()
