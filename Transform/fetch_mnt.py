import os
import numpy as np
import rioxarray
import odc.stac
import pystac_client
import planetary_computer
import matplotlib.pyplot as plt

# Import direct depuis TON architecture !
from config import SITES_PILOTES, LOGGER

# --- CONFIGURATION ---
DOSSIER_BASE = r"Outputs"

def afficher_mnt(chemin_fichier_mnt, nom_site):
    """Ouvre et affiche un fichier MNT .tif avec de belles couleurs topographiques."""
    LOGGER.info(f"📊 Préparation de l'affichage pour {nom_site}...")
    
    if not os.path.exists(chemin_fichier_mnt):
        LOGGER.error(f"❌ Le fichier est introuvable : {chemin_fichier_mnt}")
        return

    # 1. Chargement de l'image
    ds_mnt = rioxarray.open_rasterio(chemin_fichier_mnt).squeeze()
    matrice_altitude = ds_mnt.values
    
    # 2. Création de la figure
    plt.figure(figsize=(10, 8))
    
    # cmap='terrain' est la palette classique pour la cartographie d'altitude
    img_plot = plt.imshow(matrice_altitude, cmap='terrain') 
    
    # Ajout de la légende
    plt.colorbar(img_plot, fraction=0.046, pad=0.04, label="Altitude (mètres)")
    
    plt.title(f"Modèle Numérique de Terrain (30m) - {nom_site}", fontsize=15, fontweight='bold')
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def download_and_align_mnt_for_all_sites():
    LOGGER.info("========================================")
    LOGGER.info("⛰️ EXTRACTION DES MNT POUR TOUS LES SITES PILOTES")
    LOGGER.info("========================================")

    # Connexion au catalogue ESA
    LOGGER.info("🔍 Connexion au catalogue Copernicus...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    for nom_site in SITES_PILOTES.keys():
        LOGGER.info(f"\n🌍 Traitement du site : {nom_site}")
        
        dossier_tif = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}", "3_Indices", "TIF_Data")
        
        if not os.path.exists(dossier_tif):
            LOGGER.warning(f"   ⚠️ Dossier {dossier_tif} introuvable. As-tu déjà traité les images pour ce site ?")
            continue

        try:
            fichiers_dossier = os.listdir(dossier_tif)
            fichier_moule = next(f for f in fichiers_dossier if "NDVI.tif" in f)
            template_path = os.path.join(dossier_tif, fichier_moule)
        except StopIteration:
            LOGGER.warning(f"   ❌ Aucun NDVI trouvé dans {dossier_tif} pour servir de moule.")
            continue

        ds_template = rioxarray.open_rasterio(template_path).squeeze()
        crs_cible = ds_template.rio.crs
        
        bbox_exacte = ds_template.rio.transform_bounds("EPSG:4326")
        LOGGER.info(f"   📍 BBox stricte extraite du NDVI : {bbox_exacte}")

        output_mnt = os.path.join(dossier_tif, f"{nom_site}_MNT.tif")

        # 2. Recherche STAC
        search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox_exacte)
        items = list(search.items())
        
        if not items:
            LOGGER.warning("   ❌ Aucun MNT trouvé sur les serveurs pour cette zone.")
            continue

        # 3. Téléchargement direct en UTM avec des pixels de 30m
        LOGGER.info(f"   📥 Téléchargement direct dans le CRS de l'image ({crs_cible})...")
        ds_mnt_brut = odc.stac.stac_load(
            items, 
            bbox=bbox_exacte, 
            bands=["data"], 
            crs=crs_cible,       
            resolution=30,       
            patch_url=planetary_computer.sign
        ).isel(time=0)["data"]

        LOGGER.info("   📐 Alignement sub-pixel final...")
        ds_mnt_aligned = ds_mnt_brut.rio.reproject_match(ds_template)
        
        # Nettoyage des valeurs vides (NoData)
        ds_mnt_aligned = ds_mnt_aligned.astype('float32')
        ds_mnt_aligned = ds_mnt_aligned.where(ds_mnt_aligned != ds_mnt_aligned.rio.nodata, np.nan)

        n_valid_aligned = np.sum(np.isfinite(ds_mnt_aligned.values))
        LOGGER.info(f"   📊 MNT aligné : {ds_mnt_aligned.size} pixels, {n_valid_aligned} valides")

        # 4. Sauvegarde
        ds_mnt_aligned.rio.to_raster(output_mnt)
        LOGGER.info(f"   ✅ MNT sauvegardé avec succès : {output_mnt}")

if __name__ == "__main__":
    # 1. On télécharge et on aligne tous les MNT
    download_and_align_mnt_for_all_sites() 
    
    # 2. On affiche le résultat pour TOUS les sites
    LOGGER.info("\n========================================")
    LOGGER.info("🖼️ AFFICHAGE DES MNT POUR VÉRIFICATION")
    LOGGER.info("========================================")
    
    for nom_site in SITES_PILOTES.keys():
        # Construction dynamique du chemin pour chaque pays
        dossier_indices = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}", "3_Indices", "TIF_Data")
        fichier_mnt = os.path.join(dossier_indices, f"{nom_site}_MNT.tif")
        
        # Affichage
        afficher_mnt(fichier_mnt, f"Site Pilote: {nom_site}")