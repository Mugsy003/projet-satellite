import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_tif(tif_path, out_path=None):
    if not os.path.exists(tif_path):
        print(f"Erreur : Le fichier {tif_path} n'existe pas.")
        return
        
    print(f"Ouverture de {tif_path}...")
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        
    # Masquer les valeurs NaN ou très extrêmes
    data_masked = np.ma.masked_invalid(data)
    
    # Calcul des statistiques pour affichage
    mean_et = np.nanmean(data)
    max_et = np.nanmax(data)
    min_et = np.nanmin(data)
    print(f"Statistiques ET : Moyenne={mean_et:.2f}, Min={min_et:.2f}, Max={max_et:.2f}")

    # Création du plot
    plt.figure(figsize=(10, 8))
    # Echelle de couleur fixée entre 0 et 1 comme demandé
    im = plt.imshow(data_masked, cmap='YlGnBu', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Évapotranspiration (unités brutes)")
    
    filename = os.path.basename(tif_path)
    date_str = filename.split('_')[0] if '_' in filename else "Inconnue"
    plt.title(f"Carte spatiale d'Évapotranspiration - {date_str}", fontsize=14, pad=15)
    plt.axis('off') # Cacher les axes des pixels
    
    # Sauvegarde
    if out_path is None:
        out_path = tif_path.replace('.tif', '.png')
        
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Image PNG generee avec succes : {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Convertir un TIF d'ET en image PNG.")
    parser.add_argument('--input', type=str, default=None, help="Chemin vers un fichier TIF spécifique")
    parser.add_argument('--folder', type=str, default=r"C:\Users\a951444\Workspace\projet-satellite\Outputs\Serie_Temporelle_Gebesee\ET_PT_SINRH_ERA5", help="Dossier contenant les TIF (traitera le premier trouvé si --input n'est pas fourni)")
    
    args = parser.parse_args()
    
    if args.input:
        plot_tif(args.input)
    else:
        # Chercher le premier TIF dans le dossier
        files = glob.glob(os.path.join(args.folder, "*.tif"))
        if not files:
            print(f"Aucun fichier TIF trouvé dans {args.folder}")
            return
            
        # Prendre par exemple une belle journée d'été si possible (ex: Juillet/Août)
        summer_files = [f for f in files if "-07-" in f or "-08-" in f]
        target_file = summer_files[0] if summer_files else files[0]
        
        plot_tif(target_file)

if __name__ == "__main__":
    main()
