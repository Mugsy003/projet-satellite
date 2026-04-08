import os
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import IsolationForest
import tkinter as tk
from tkinter import simpledialog
from scipy.ndimage import median_filter as scipy_median



def save_comparative_band_curves(mes_images_arrays,output_dir="Comparaisons"):
    """
    Génère et sauvegarde les courbes de distribution des pixels pour chaque bande (R, G, B),
    en comparant tous les pays sur le même graphique.
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    bandes_info = {
        0: {"nom": "Rouge", "couleur_titre": "red"},
        1: {"nom": "Verte", "couleur_titre": "green"},
        2: {"nom": "Bleue", "couleur_titre": "blue"}
    }

    for band_idx, info in bandes_info.items():
        plt.figure(figsize=(10, 6))
        
        for pays, img_reflectance in mes_images_arrays.items():
            # Extraire la bande et aplatir en 1D
            band_data = img_reflectance[band_idx].ravel()
            # Retirer les pixels masqués (NaN)
            band_data = band_data[~np.isnan(band_data)]
            
            # Tracer une courbe (histogramme vide avec contours)
            # density=True est crucial : cela permet de comparer des pays qui n'ont 
            # pas la même taille (pas le même nombre de pixels total)
            plt.hist(band_data, bins=100, histtype='step', linewidth=2.5, 
                     density=True, label=pays)

        plt.title(f"Distribution des pixels - Bande {info['nom']} (Comparaison)", fontsize=14, color=info['couleur_titre'])
        plt.xlabel("Valeur de Réflectance")
        plt.ylabel("Densité (Proportion de pixels)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Sauvegarde au lieu d'afficher
        filepath = os.path.join(output_dir, f"Comparaison_Bande_{info['nom']}.png")
        plt.savefig(filepath, dpi=300)
        plt.close() 
        
    return f"✅ Courbes comparatives sauvegardées dans le dossier '{output_dir}/'"


def plot_reflectance_histograms(img_reflectance, pays):
    """
    Génère et affiche les histogrammes de réflectance pour les bandes RGB.
    Aucun clipping n'est appliqué pour observer la distribution brute.
    
    Args:
        img_reflectance (np.ndarray): Array de forme (3, H, W) correspondant à (Red, Green, Blue).
        pays (str): Nom du pays pour le titre du graphique.
    """
    # Aplatir les tableaux en 1D
    red = img_reflectance[0].ravel()
    green = img_reflectance[1].ravel()
    blue = img_reflectance[2].ravel()

    # Retirer les NaN (qui correspondent aux bords noirs ou aux nuages masqués)
    red = red[~np.isnan(red)]
    green = green[~np.isnan(green)]
    blue = blue[~np.isnan(blue)]

    plt.figure(figsize=(10, 5))
    
    # Création des histogrammes (bins=100 pour une bonne précision)
    plt.hist(red, bins=100, color='red', alpha=0.5, label='Bande Rouge')
    plt.hist(green, bins=100, color='green', alpha=0.5, label='Bande Verte')
    plt.hist(blue, bins=100, color='blue', alpha=0.5, label='Bande Bleue')

    plt.title(f"Distribution des pixels - {pays}")
    plt.xlabel("Valeur de Réflectance")
    plt.ylabel("Nombre de pixels")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig(f"Histogrammes/composition_rgb_{pays}.png", dpi=300, bbox_inches='tight')
    plt.close()

def get_bbox_from_point(lon, lat, radius_km=30):
    """
    Calcule une Bounding Box de 'radius_km' autour d'un point GPS.
    Retourne [min_lon, min_lat, max_lon, max_lat]
    """
    # 1 degré de latitude ~ 111.32 km
    lat_delta = radius_km / 111.32
    # 1 degré de longitude dépend de la latitude (cosinus)
    lon_delta = radius_km / (111.32 * math.cos(math.radians(lat)))
    
    return [lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta]

def landsat_dn_to_reflectance(dn_array):
    """
    Convertit les Digital Numbers (DN) de Landsat Collection 2 Level 2 
    en valeurs de réflectance de surface (0.0 - 1.0).
    """
    reflectance = (dn_array * 0.0000275) - 0.2
    
    return reflectance

def get_landsat_mask(qa_values):
    qa_int = qa_values.astype(int)
    cloud = (qa_int >> 3) & 1
    shadow = (qa_int >> 4) & 1
    dilated = (qa_int >> 1) & 1
    cirrus = (qa_int >> 2) & 1
    
    mask = ((cloud == 0) & (shadow == 0) & (dilated == 0) & (cirrus == 0)).astype(np.uint8)
    return mask

def median_filter_2d(dn_array, kernel_size=3):
    filtered = np.empty_like(dn_array)
    for i in range(dn_array.shape[0]):
        band = dn_array[i].copy()
        mask = np.isnan(band)
        
        # Astuce : On remplace temporairement les NaN par la médiane de la bande
        if np.any(~mask):
            fill_value = np.nanmedian(band)
            band_filled = np.nan_to_num(band, nan=fill_value)
            
            f_band = scipy_median(band_filled, size=kernel_size)
            f_band[mask] = np.nan # On remet les trous d'origine
            filtered[i] = f_band
        else:
            filtered[i] = np.nan
            
    return filtered

def stretch_z_score(dn_array, z_min=-2.0, z_max=2.0):
    stretched = np.empty_like(dn_array)
    for i in range(dn_array.shape[0]):
        band = dn_array[i]
        mean, std = np.nanmean(band), np.nanstd(band)
        
        if std == 0 or np.isnan(std):
            stretched[i] = band
            continue
            
        z_score = (band - mean) / std
        # On clip et on normalise entre 0 et 1
        z_clipped = np.clip(z_score, z_min, z_max)
        stretched[i] = (z_clipped - z_min) / (z_max - z_min)
        
    return stretched

def stretch_iqr(dn_array):
    # Calcul des quartiles globaux pour l'image
    q1 = np.nanpercentile(dn_array, 25)
    q3 = np.nanpercentile(dn_array, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Sécurité division par zéro
    if np.isnan(lower_bound) or np.isnan(upper_bound) or (upper_bound == lower_bound):
        return dn_array
    
    stretched = np.clip(dn_array, lower_bound, upper_bound)
    return (stretched - lower_bound) / (upper_bound - lower_bound)



def isolation_forest_filter(dn_array, contamination=0.05):
    filtered = np.empty_like(dn_array)
    b, h, l = dn_array.shape
    
    # Aplatissement pour le ML : (Pixels, Bandes)
    array_reshaped = dn_array.transpose(1, 2, 0)
    pixels_flat = array_reshaped.reshape(-1, b)
    
    mask_valid = ~np.isnan(pixels_flat).any(axis=1)
    
    if np.any(mask_valid):
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        predictions = iso.fit_predict(pixels_flat[mask_valid])
        
        final_mask_flat = np.full(pixels_flat.shape[0], 1)
        final_mask_flat[mask_valid] = predictions
        mask_2d = final_mask_flat.reshape(h, l)
        
        for i in range(b):
            band = dn_array[i].copy()
            band[mask_2d == -1] = np.nan # Supprime les outliers
            filtered[i] = band
    else:
        filtered[:] = np.nan
        
    return filtered


def compter_pourcentage_blancs(image_rgb):
    """
    Calcule le pourcentage de pixels 100% blancs dans une image RGB.
    Accepte les images en entier (0-255) et en float (0.0-1.0).
    L'image doit être au format (Hauteur, Largeur, 3).
    """
    # 1. Déterminer ce qu'est le "blanc" selon le type de ton image
    # Si c'est une image en réflectance (float entre 0 et 1)
    if image_rgb.dtype.kind == 'f': 
        valeur_blanc = 1.0
    # Si c'est une image classique (entiers de 0 à 255)
    else: 
        valeur_blanc = 255

    # 2. Trouver les pixels où (Rouge == blanc) ET (Vert == blanc) ET (Bleu == blanc)
    # axis=-1 permet de vérifier la dernière dimension (les 3 canaux de couleur en même temps)
    masque_blanc = np.all(image_rgb == valeur_blanc, axis=-1)

    # 3. Compter combien de fois c'est Vrai (True = 1)
    nombre_de_blancs = np.sum(masque_blanc) 
    pourcentage_blancs = (nombre_de_blancs / (image_rgb.shape[0] * image_rgb.shape[1])) * 100
    
    return pourcentage_blancs

def serie_temporelle_pourcentage_nuages(img_rgb):
    """
    Calcule la série temporelle du pourcentage de pixels nuageux dans une image RGB.
    L'image doit être au format (Temps, Hauteur, Largeur, 3).
    On considère qu'un pixel est nuageux s'il est 100% blanc.
    """
    pourcentages_nuages = []
    
    for t in range(img_rgb.shape[0]):
        pourcentage = compter_pourcentage_blancs(img_rgb[t])
        pourcentages_nuages.append(pourcentage)
    
    return pourcentages_nuages


def demander_aoi_popup():
    """
    Ouvre une fenêtre pop-up pour demander les coordonnées GPS à l'utilisateur.
    Retourne un tuple (longitude, latitude) ou None si annulé.
    """
    # 1. Créer la fenêtre principale Tkinter
    root = tk.Tk()
    root.withdraw()
    
    # 2. Ouvrir la boîte de dialogue
    reponse = simpledialog.askstring(
        titre="Choix de la Zone d'Intérêt (AOI)",
        invite="Entrez les coordonnées (Longitude, Latitude) séparées par une virgule :\nExemple pour Paris : 2.35, 48.85"
    )
    
    # 3. Traiter la réponse
    if reponse:
        try:
            lon_str, lat_str = reponse.split(",")
            lon = float(lon_str.strip())
            lat = float(lat_str.strip())
            return (lon, lat)
        except ValueError:
            print("⚠️ Erreur : Le format n'est pas respecté (attendu : lon, lat).")
            return None
    else:
        return None