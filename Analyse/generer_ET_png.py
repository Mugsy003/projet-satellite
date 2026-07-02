import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Dossier de base
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Outputs")

def generer_png_pour_tif(tif_path):
    """Génère un PNG coloré à partir d'un fichier TIF d'ET."""
    # Le fichier PNG sera sauvegardé au même endroit, avec l'extension .png
    png_path = tif_path.replace('.tif', '_map.png')
    
    # Éviter de recréer si déjà existant
    if os.path.exists(png_path):
        return
        
    try:
        with rasterio.open(tif_path) as src:
            et_data = src.read(1)
            
            # Masquer les valeurs invalides (NoData)
            et_data = np.where(et_data < 0, np.nan, et_data)
            
            # Vérifier s'il y a des données valides
            if np.all(np.isnan(et_data)):
                return
                
            plt.figure(figsize=(10, 8))
            
            # Utilisation du 95ème percentile pour ne pas être faussé par les valeurs extrêmes
            vmax = np.nanpercentile(et_data, 95)
            if np.isnan(vmax) or vmax <= 0:
                vmax = 1.0  # Valeur par défaut si tout est à 0
                
            plt.imshow(et_data, cmap='YlGnBu', vmin=0, vmax=vmax)
            plt.colorbar(label='Évapotranspiration (mm/h)')
            
            # Extraction du nom du site et de la date depuis le nom du fichier
            basename = os.path.basename(tif_path)
            # Format attendu : YYYY-MM-DD_Site_ET_mm_h.tif
            parts = basename.split('_')
            date_str = parts[0]
            site_str = parts[1] if len(parts) > 1 else "Inconnu"
            
            source = "ERA5" if "ERA5" in tif_path else "ICOS"
            
            plt.title(f"Carte ET - {site_str} ({date_str}) [{source}]")
            plt.axis('off')
            
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"Erreur lors de la génération de la carte pour {os.path.basename(tif_path)} : {e}")


def main():
    print("=" * 60)
    print("🌍 GÉNÉRATION DES CARTES D'ÉVAPOTRANSPIRATION (PNG)")
    print("=" * 60)
    
    # Rechercher tous les fichiers TIF d'ET (ICOS et ERA5)
    pattern_icos = os.path.join(OUTPUTS_DIR, "Serie_Temporelle_*", "ET_TTME", "*_ET_mm_h.tif")
    pattern_era5 = os.path.join(OUTPUTS_DIR, "Serie_Temporelle_*", "ET_TTME_ERA5", "*_ET_mm_h.tif")
    
    tifs = glob.glob(pattern_icos) + glob.glob(pattern_era5)
    
    if not tifs:
        print("Aucun fichier TIF d'ET trouvé.")
        return
        
    print(f"📁 {len(tifs)} fichiers TIF trouvés. Génération en cours...")
    
    count = 0
    for i, tif in enumerate(tifs):
        png_path = tif.replace('.tif', '_map.png')
        if not os.path.exists(png_path):
            generer_png_pour_tif(tif)
            count += 1
            if count % 10 == 0:
                print(f"   ... {i+1}/{len(tifs)} traités")
                
    print(f"✅ Terminé ! {count} nouvelles cartes PNG générées.")
    print("Les fichiers PNG se trouvent dans les mêmes dossiers que les fichiers TIF originaux.")

if __name__ == "__main__":
    import sys
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
