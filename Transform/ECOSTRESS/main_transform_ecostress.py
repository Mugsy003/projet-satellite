"""
Transform/main_transform_ecostress.py
Visualisation thermique dédiée pour ECOSTRESS (pas de RGB optique).
"""
import os
import json
import numpy as np
import pystac_client
import rasterio

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import LOGGER, SITES_PILOTES, OUTPUT_DIR, radius_km
from Utils import get_bbox_from_point

import Transform.ECOSTRESS.processor_ecostress as processor_eco


def save_ecostress_lst_maps(liste_images, pays, output_dir):
    """
    Sauvegarde les cartes LST ECOSTRESS :
      - PNG avec colormap inferno et bornes adaptatives
      - GeoTIFF dans TIF_Data/
    """
    dossier_base = os.path.join(output_dir, f"Serie_Temporelle_{pays}")
    dossier_lst = os.path.join(dossier_base, "LST_Maps")
    dossier_tif = os.path.join(dossier_base, "TIF_Data")
    os.makedirs(dossier_lst, exist_ok=True)
    os.makedirs(dossier_tif, exist_ok=True)

    for item in liste_images:
        date_str = item["date"]
        lst = item["lst_celsius"]

        # --- Bornes adaptatives basées sur les percentiles réels ---
        vmin = np.nanpercentile(lst, 2)
        vmax = np.nanpercentile(lst, 98)
        # Assurer un écart minimum pour le contraste
        if vmax - vmin < 3:
            mid = (vmin + vmax) / 2
            vmin, vmax = mid - 2, mid + 2

        lst_mean = np.nanmean(lst)
        lst_std = np.nanstd(lst)
        nb_valid = np.count_nonzero(~np.isnan(lst))

        # --- PNG : carte LST colorée ---
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(lst, cmap="inferno", vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Température de surface (°C)", fontsize=12)
        ax.set_title(
            f"ECOSTRESS LST - {pays}\n{date_str}  |  "
            f"Moy: {lst_mean:.1f}°C  σ: {lst_std:.1f}°C  "
            f"({nb_valid} px valides)",
            fontsize=13
        )
        ax.axis("off")

        nom_png = f"{date_str}_{pays}_LST.png"
        plt.savefig(os.path.join(dossier_lst, nom_png), dpi=200, bbox_inches='tight')
        plt.close(fig)

        # --- GeoTIFF ---
        nom_tif = f"{date_str}_{pays}_LST.tif"
        chemin_tif = os.path.join(dossier_tif, nom_tif)
        hauteur, largeur = lst.shape

        with rasterio.open(
            chemin_tif, 'w',
            driver='GTiff',
            height=hauteur, width=largeur,
            count=1,
            dtype='float32',
            nodata=np.nan,
            transform=item["transform"],
            crs=item["crs"]
        ) as dst:
            dst.write(lst.astype('float32'), 1)

    LOGGER.info(f"      ✅ {len(liste_images)} cartes LST sauvegardées dans : {dossier_lst}")
    LOGGER.info(f"      ✅ {len(liste_images)} GeoTIFF sauvegardés dans : {dossier_tif}")


def main():
    LOGGER.info("🚀 DÉBUT DE LA PHASE 2 ECOSTRESS : TRANSFORMATION")
    
    chemin_manifeste = os.path.join(OUTPUT_DIR, "manifest_extraction_ecostress.json")
    
    if not os.path.exists(chemin_manifeste):
        LOGGER.error(f"❌ Impossible de trouver le manifeste : {chemin_manifeste}")
        LOGGER.error("Veuillez exécuter 'python -m Extraction.main_extract_ecostress' en premier.")
        return

    with open(chemin_manifeste, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    LOGGER.info("🌐 Reconnexion au catalogue STAC NASA CMR (LPCLOUD)...")
    catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")

    for pays, liste_ids in manifest_data.items():
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"⚙️ Transformation ECOSTRESS pour le site : {pays}")
        
        if not liste_ids:
            LOGGER.warning(f"   ⚠️ Aucun ID trouvé dans le manifeste pour {pays}.")
            continue
            
        coords = SITES_PILOTES[pays]
        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km=radius_km)
        
        import time
        from pystac_client.exceptions import APIError
        search = catalog.search(collections=["ECO_L2T_LSTE.v002"], ids=liste_ids)
        
        max_retries = 5
        mes_items = None
        for attempt in range(max_retries):
            try:
                mes_items = list(search.items())
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
                
        if not mes_items:
            continue
        
        liste_images = processor_eco.process_ecostress_timeseries(
            mes_items=mes_items, 
            bbox=bbox
        )
        
        if not liste_images:
            continue
            
        # Visualisation thermique dédiée (pas de RGB optique)
        save_ecostress_lst_maps(liste_images, pays + "_ECOSTRESS", OUTPUT_DIR)
    
    LOGGER.info("\n✅ PHASE 2 ECOSTRESS TERMINÉE. Données transformées avec succès !")

def run():
    # Configuration de l'authentification EarthData pour GDAL
    home = os.path.expanduser("~")
    cookie_file = os.path.join(home, ".urs_cookies")
    netrc_file = os.path.join(home, "_netrc")
    
    os.environ["GDAL_HTTP_COOKIEFILE"] = cookie_file
    os.environ["GDAL_HTTP_COOKIEJAR"] = cookie_file
    os.environ["GDAL_HTTP_NETRC"] = "YES"
    os.environ["GDAL_HTTP_NETRC_FILE"] = netrc_file
    
    LOGGER.info(f"🔐 Auth EarthData configurée (netrc: {netrc_file})")
    
    with rasterio.Env(
        GDAL_HTTP_MAX_RETRY=10,       
        GDAL_HTTP_RETRY_DELAY=30,      
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
        GDAL_HTTP_TIMEOUT=60,
        GDAL_HTTP_COOKIEFILE=cookie_file,
        GDAL_HTTP_COOKIEJAR=cookie_file,
    ):
        main()

if __name__ == "__main__":
    run()

