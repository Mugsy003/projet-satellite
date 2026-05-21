"""
Script brouillon : collecte toutes les images de comparaison (DMS, TsHARP, Fusion)
de tous les sites pilotes et les copie dans un seul dossier.
"""
import os
import shutil
import glob
from config import SITES_PILOTES

DOSSIER_BASE = "Outputs"
DOSSIER_DEST = os.path.join("Outputs", "Toutes_Comparaisons")
os.makedirs(DOSSIER_DEST, exist_ok=True)

compteur = 0

for nom_site in SITES_PILOTES.keys():
    # Dossier Landsat
    dossier_landsat = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}", "3_Indices", "TIF_Data")
    # Dossier S2
    dossier_s2 = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")

    for dossier in [dossier_landsat, dossier_s2]:
        if not os.path.exists(dossier):
            continue
        # Chercher tous les PNG de comparaison
        pngs = glob.glob(os.path.join(dossier, "*Comparaison*.png"))
        for src in pngs:
            nom_fichier = os.path.basename(src)
            dest = os.path.join(DOSSIER_DEST, nom_fichier)
            shutil.copy2(src, dest)
            compteur += 1

print(f"\n[OK] {compteur} images copiees dans : {DOSSIER_DEST}")
