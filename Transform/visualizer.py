"""
Transform/visualizer.py
Gère l'étirement du contraste, l'application des filtres de lissage
et la sauvegarde des images finales (.png).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from config import LOGGER
from Utils import stretch_iqr, median_filter_2d, save_comparative_band_curves

def process_and_save_filtered_composition(img_reflectance, pays, output_dir):
    """Applique un filtre médian, étire le contraste (IQR) et sauvegarde l'image."""
    LOGGER.info(f"   -> Traitement visuel (Filtre Médian + IQR) pour {pays}...")
    
    # 1. Application du filtre médian pour lisser le bruit poivre/sel (desactivée pour l'instant)
    img_median = median_filter_2d(img_reflectance, kernel_size=3)
    
    # 2. Transposition (Hauteur, Largeur, Couleurs) et étirement du contraste
    img_rgb = img_reflectance.transpose(1, 2, 0)
    img_rgb = stretch_iqr(img_rgb)
    
    # Sécurité pour Matplotlib (valeurs entre 0 et 1)
    img_rgb_clipped = np.clip(img_rgb, 0.0, 1.0)
    
    # 3. Sauvegarde de l'image
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb_clipped)
    plt.title(f"Composition RGB Landsat + Filtrage ({pays})", fontsize=14)
    plt.axis("off")
    
    chemin_sauvegarde = os.path.join(output_dir, f"{pays}_RGB_Filtre.png")
    plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
    plt.close() # Libère la mémoire RAM
    
    LOGGER.info(f"      ✅ Carte sauvegardée : {chemin_sauvegarde}")

def generate_comparative_histograms(mes_images_arrays, output_dir):
    """Génère les graphiques de distribution de réflectance entre tous les pays."""
    LOGGER.info("\n📊 Génération des courbes comparatives de réflectance...")
    dossier = save_comparative_band_curves(mes_images_arrays, output_dir=output_dir)
    LOGGER.info(f"   ✅ Graphiques comparatifs sauvegardés dans : {dossier}")

def save_raw_reference_image(img_rgb_brute, pays, output_dir):
    """Sauvegarde l'image brute de référence avant tout masquage ou filtrage."""
    LOGGER.info(f"   -> Sauvegarde de l'image brute de référence pour {pays}...")
    
    # Étirement du contraste indispensable pour l'œil humain
    p_bas, p_haut = np.nanpercentile(img_rgb_brute, (2, 98))
    img_rgb_etiree = (img_rgb_brute - p_bas) / (p_haut - p_bas)
    img_rgb_etiree = np.clip(img_rgb_etiree, 0.0, 1.0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb_etiree)
    plt.title(f"Image Brute de Référence - {pays}", fontsize=14)
    plt.axis("off")
    
    # On ajoute "0_" dans le nom pour qu'elle apparaisse en premier dans ton dossier
    chemin_sauvegarde = os.path.join(output_dir, f"{pays}_0_Image_Brute.png")
    plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
    plt.close()

def plot_images_selectionnees(data_cube, mes_items, pays, output_dir=None):
    """
    Affiche (ou sauvegarde) les images brutes contenues dans le cube de données,
    les unes à côté des autres, avec leur date et le nom du satellite.
    """
    nb_images = data_cube.sizes['time']
    
    fig, axes = plt.subplots(1, nb_images, figsize=(6 * nb_images, 6))
    if nb_images == 1:
        axes = [axes]
        
    for i in range(nb_images):
        # 1. Extraction des bandes
        rouge = data_cube["red"].isel(time=i).values
        vert = data_cube["green"].isel(time=i).values
        bleu = data_cube["blue"].isel(time=i).values
        
        # 2. Création de l'image RGB
        hauteur, largeur = rouge.shape
        img_rgb = np.zeros((hauteur, largeur, 3))
        img_rgb[:, :, 0] = rouge
        img_rgb[:, :, 1] = vert
        img_rgb[:, :, 2] = bleu
        
        # 3. Étirement du contraste
        p_bas, p_haut = np.nanpercentile(img_rgb, (2, 98))
        img_rgb_etiree = (img_rgb - p_bas) / (p_haut - p_bas)
        img_rgb_etiree = np.clip(img_rgb_etiree, 0.0, 1.0)
        
        # 4. Extraction de la date ET du nom du satellite
        date_str = str(data_cube.time.values[i])[:10]
        
        # On récupère l'objet STAC d'origine pour lire le nom du satellite (platform)
        if i < len(mes_items):
            nom_satellite = mes_items[i].properties.get('platform', 'Landsat')
        else:
            nom_satellite = "Landsat"
            
        # 5. Affichage avec le titre sur deux lignes (\n)
        axes[i].imshow(img_rgb_etiree)
        axes[i].set_title(f"{date_str}\n({nom_satellite.upper()})", fontsize=14, fontweight='bold')
        axes[i].axis('off')
        
    plt.suptitle(f"Images sources sélectionnées pour {pays}", fontsize=18)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        chemin = os.path.join(output_dir, f"{pays}_Images_Sources.png")
        plt.savefig(chemin, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_timeseries_images(liste_images, pays, output_dir):
    """
    Parcourt la liste des images temporelles générées et les sauvegarde
    avec leur date dans le nom du fichier.
    """
    LOGGER.info(f"   -> Sauvegarde de la série temporelle RGB pour {pays}...")
    
    # On crée un sous-dossier par pays pour que ce soit propre
    dossier_pays = os.path.join(output_dir, f"Sery_Temporelle_{pays}")
    os.makedirs(dossier_pays, exist_ok=True)
    
    for item in liste_images:
        date_str = item["date"]
        img_reflectance = item["reflectance"]
        nb_fusion = item["nb_images_fusionnees"]
        
        # Préparation RGB avec étirement du contraste
        img_rgb = img_reflectance.transpose(1, 2, 0)
        img_rgb = stretch_iqr(img_rgb)
        img_rgb_clipped = np.clip(img_rgb, 0.0, 1.0)
        
        # Sauvegarde
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb_clipped)
        
        # Titre dynamique selon s'il y a eu réparation ou non
        if nb_fusion == 1:
            titre = f"Image Nette et Parfaite - {pays} ({date_str})"  
            suffixe = "Parfaite"
        else:
            titre = f"Image Réparée ({nb_fusion} fusions) - {pays} ({date_str})"
            suffixe = "Reparee"
            
        plt.title(titre, fontsize=14)
        plt.axis("off")
        
        nom_fichier = f"{date_str}_{pays}_{suffixe}.png"
        chemin_sauvegarde = os.path.join(dossier_pays, nom_fichier)
        
        plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
        plt.close()
        
    LOGGER.info(f"      ✅ {len(liste_images)} images sauvegardées dans : {dossier_pays}")

def save_timeseries_results(liste_images, pays, output_dir):
    """
    Parcourt la liste des images temporelles et sauvegarde l'image BRUTE 
    et l'image TRAITÉE (Réflectance) pour comparer.
    """
    LOGGER.info(f"   -> Sauvegarde des comparaisons Avant/Après pour {pays}...")
    
    # Création de sous-dossiers propres
    dossier_brutes = os.path.join(output_dir, f"Serie_Temporelle_{pays}", "1_Brutes")
    dossier_traitees = os.path.join(output_dir, f"Serie_Temporelle_{pays}", "2_Traitees")
    os.makedirs(dossier_brutes, exist_ok=True)
    os.makedirs(dossier_traitees, exist_ok=True)
    
    for item in liste_images:
        date_str = item["date"]
        img_reflectance = item["reflectance"]
        img_brute = item["brute"]
        nb_fusion = item["nb_images_fusionnees"]
        
        # --- 1. SAUVEGARDE DE L'IMAGE BRUTE ---
        p_bas, p_haut = np.nanpercentile(img_brute, (2, 98))
        img_rgb_brute = (img_brute - p_bas) / (p_haut - p_bas)
        img_rgb_brute = np.clip(img_rgb_brute, 0.0, 1.0)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb_brute)
        plt.title(f"Image Brute (Avant traitement) - {pays} ({date_str})", fontsize=14)
        plt.axis("off")
        plt.savefig(os.path.join(dossier_brutes, f"{date_str}_{pays}_Brute.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # --- 2. SAUVEGARDE DE L'IMAGE TRAITÉE ---
        img_rgb_traitee = img_reflectance.transpose(1, 2, 0)
        img_rgb_traitee = stretch_iqr(img_rgb_traitee)
        img_rgb_traitee = np.clip(img_rgb_traitee, 0.0, 1.0)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img_rgb_traitee)
        
        etat = item.get("etat", "Inconnu")
        
        if etat == "Parfaite":
            titre = f"Image Nette (Parfaite) - {pays} ({date_str})"
            suffixe = "Parfaite"
        elif etat == "Reparee":
            titre = f"Image Réparée ({nb_fusion} fusions) - {pays} ({date_str})"
            suffixe = "Reparee"
        else: 
            titre = f"Image Incomplète (Échec réparation) - {pays} ({date_str})"
            suffixe = "Incomplette"
            
        plt.title(titre, fontsize=14)
        plt.axis("off")
        plt.savefig(os.path.join(dossier_traitees, f"{date_str}_{pays}_{suffixe}.png"), dpi=300, bbox_inches='tight')
        plt.close()



def save_indices_maps(liste_images, pays, output_dir):
    """
    Sauvegarde les indices : 
    - Les .png dans le dossier '3_Indices' (pour la visualisation)
    - Les .tif dans le dossier '3_Indices/TIF_Data' (pour les calculs IA/DMS)
    """
    LOGGER.info(f"   -> Sauvegarde double (PNG + TIF) pour {pays}...")
    
    # 1. Dossier principal pour les images
    dossier_indices = os.path.join(output_dir, f"Serie_Temporelle_{pays}", "3_Indices")
    # 2. Sous-dossier spécifique pour les données brutes TIF
    dossier_tif = os.path.join(dossier_indices, "TIF_Data")
    
    os.makedirs(dossier_indices, exist_ok=True)
    os.makedirs(dossier_tif, exist_ok=True)
    
    cmap_settings = {
        "NDVI": {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.8, "label": "Végétation"},
        "NDWI": {"cmap": "RdBu",   "vmin": -0.5, "vmax": 0.5, "label": "Eau / Humidité"},
        "NDBI": {"cmap": "YlOrRd", "vmin": -0.5, "vmax": 0.5, "label": "Sols nus / Bâti"},
        "EVI":  {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.8, "label": "Végétation Dense"},
        "SAVI": {"cmap": "RdYlGn", "vmin": -0.2, "vmax": 0.8, "label": "Végétation (Sol aride)"},
        "LST":  {"cmap": "inferno", "vmin": 10, "vmax": 55, "label": "Température (°C)"},
        "Thermique_B10": {"cmap": "inferno", "vmin": 10, "vmax": 55, "label": "Température Brute (°C)"}
    }
    
    for item in liste_images:
        date_str = item["date"]
        indices_dict = item["indices"]
        
        for nom_indice, matrice in indices_dict.items():
            settings = cmap_settings[nom_indice]
            
            # --- A. SAUVEGARDE DU PNG (Visualisation) ---
            plt.figure(figsize=(10, 10))
            img_plot = plt.imshow(matrice, cmap=settings["cmap"], vmin=settings["vmin"], vmax=settings["vmax"])
            plt.colorbar(img_plot, fraction=0.046, pad=0.04, label=f"Valeur {nom_indice}")
            plt.title(f"Carte {nom_indice} - {pays} ({date_str})", fontsize=14)
            plt.axis("off")
            
            nom_png = f"{date_str}_{pays}_{nom_indice}.png"
            plt.savefig(os.path.join(dossier_indices, nom_png), dpi=300, bbox_inches='tight')
            plt.close()

            # --- B. SAUVEGARDE DU TIF (Données brutes pour l'IA) ---
            nom_tif = f"{date_str}_{pays}_{nom_indice}.tif"
            chemin_tif = os.path.join(dossier_tif, nom_tif)
            
            hauteur, largeur = matrice.shape
            
            with rasterio.open(
                chemin_tif, 'w',
                driver='GTiff',
                height=hauteur, width=largeur,
                count=1,
                dtype=matrice.dtype,
                nodata=np.nan,
                transform=item["transform"],
                crs=item["crs"] 
            ) as dst:
                dst.write(matrice, 1)

    LOGGER.info(f"      ✅ PNG sauvegardés dans : 3_Indices")
    LOGGER.info(f"      ✅ TIF sauvegardés dans : 3_Indices/TIF_Data")