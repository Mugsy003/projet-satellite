import os
import glob
import shutil
from config import SITES_PILOTES

for site in SITES_PILOTES.keys():
    folder = f"Outputs/Serie_Temporelle_{site}_S3"
    tif_brutes = glob.glob(os.path.join(folder, "TIF_Data", "*.tif"))
    for tif in tif_brutes:
        os.remove(tif)
    print(f"Supprimé {len(tif_brutes)} fichierstif dans {folder}/1_Brutes")