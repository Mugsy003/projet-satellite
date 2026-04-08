import os
import numpy as np
import matplotlib.pyplot as plt

def save_comparative_band_curves(mes_images_arrays, output_dir="Comparaisons"):
    """
    Génère et sauvegarde les courbes de distribution des pixels pour chaque bande (R, G, B),
    en comparant tous les pays sur le même graphique.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    bandes_info = {
        0: {"nom": "Rouge", "couleur_titre": "red"},
        1: {"nom": "Verte", "couleur_titre": "green"},
        2: {"nom": "Bleue", "couleur_titre": "blue"}
    }

    for band_idx, info in bandes_info.items():
        plt.figure(figsize=(10, 6))
        
        for pays, img_reflectance in mes_images_arrays.items():
            band_data = img_reflectance[band_idx].ravel()
            band_data = band_data[~np.isnan(band_data)]
            
            plt.hist(band_data, bins=100, histtype='step', linewidth=2.5, 
                     density=True, label=pays)

        plt.title(f"Distribution des pixels - Bande {info['nom']} (Comparaison)", fontsize=14, color=info['couleur_titre'])
        plt.xlabel("Valeur de Réflectance")
        plt.ylabel("Densité (Proportion de pixels)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, f"Comparaison_Bande_{info['nom']}.png")
        plt.savefig(filepath, dpi=300)
        plt.close() 
        
    return f"✅ Courbes comparatives sauvegardées dans le dossier '{output_dir}/'"

def plot_reflectance_histograms(img_reflectance, pays):
    """
    Génère et sauvegarde les histogrammes de réflectance pour les bandes RGB.
    Aucun clipping n'est appliqué pour observer la distribution brute.
    """
    red = img_reflectance[0].ravel()
    green = img_reflectance[1].ravel()
    blue = img_reflectance[2].ravel()

    red = red[~np.isnan(red)]
    green = green[~np.isnan(green)]
    blue = blue[~np.isnan(blue)]

    plt.figure(figsize=(10, 5))
    
    plt.hist(red, bins=100, color='red', alpha=0.5, label='Bande Rouge')
    plt.hist(green, bins=100, color='green', alpha=0.5, label='Bande Verte')
    plt.hist(blue, bins=100, color='blue', alpha=0.5, label='Bande Bleue')

    plt.title(f"Distribution des pixels - {pays}")
    plt.xlabel("Valeur de Réflectance")
    plt.ylabel("Nombre de pixels")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs("Histogrammes", exist_ok=True)
    plt.savefig(f"Histogrammes/composition_rgb_{pays}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_images_selectionnees(data_cube, pays, output_dir=None):
    """
    Affiche (ou sauvegarde) les images brutes contenues dans le cube de données,
    les unes à côté des autres, avec leur date dans le titre.
    """
    # On compte combien d'images on a dans le cube (généralement 3, sauf pour le Portugal)
    nb_images = data_cube.sizes['time']
    
    # On crée une figure large avec 1 ligne et 'nb_images' colonnes
    fig, axes = plt.subplots(1, nb_images, figsize=(6 * nb_images, 6))
    
    # Sécurité : si on n'a qu'une seule image, axes n'est pas une liste, on le force en liste
    if nb_images == 1:
        axes = [axes]
        
    for i in range(nb_images):
        # 1. Extraction des bandes pour l'image 'i'
        rouge = data_cube["red"].isel(time=i).values
        vert = data_cube["green"].isel(time=i).values
        bleu = data_cube["blue"].isel(time=i).values
        
        # 2. Création de l'image RGB (Méthode explicite sans dstack)
        hauteur, largeur = rouge.shape
        img_rgb = np.zeros((hauteur, largeur, 3))
        img_rgb[:, :, 0] = rouge
        img_rgb[:, :, 1] = vert
        img_rgb[:, :, 2] = bleu
        
        # 3. Étirement du contraste (sinon l'image est noire)
        p_bas, p_haut = np.nanpercentile(img_rgb, (2, 98))
        img_rgb_etiree = (img_rgb - p_bas) / (p_haut - p_bas)
        img_rgb_etiree = np.clip(img_rgb_etiree, 0.0, 1.0)
        
        # 4. Extraction de la date propre (format YYYY-MM-DD)
        # data_cube.time.values[i] ressemble à '2025-06-25T10:30:00.0000', on garde les 10 premiers caractères
        date_str = str(data_cube.time.values[i])[:10]
        
        # 5. Affichage dans la sous-fenêtre correspondante
        axes[i].imshow(img_rgb_etiree)
        axes[i].set_title(f"{date_str}", fontsize=14, fontweight='bold')
        axes[i].axis('off')
        
    # Titre global au-dessus de toutes les images
    plt.suptitle(f"Images sources sélectionnées pour {pays}", fontsize=18)
    plt.tight_layout()
    
    # 6. Sauvegarde ou affichage
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        chemin = os.path.join(output_dir, f"{pays}_Images_Sources.png")
        plt.savefig(chemin, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()