import sys
import os
import glob
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pyproj import Transformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import OUTPUT_DIR, SITES_PILOTES, radius_km
from Traitement.LSTM_Forecasting.dataset_prep import get_homogeneous_points

def plot_spatial_maps(site="Gebesee"):
    print(f"Génération des cartes pour {site}...")
    coords = SITES_PILOTES[site]
    lon, lat = coords["lon"], coords["lat"]
    
    out_dir_et = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "Cartes_Spatiales", "ET_Predite")
    out_dir_pix = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "Cartes_Spatiales", "Pixels_Entrainement")
    os.makedirs(out_dir_et, exist_ok=True)
    os.makedirs(out_dir_pix, exist_ok=True)
    
    # Trouver les dates communes entre NDVI et ET PT-SINRH
    ndvi_files = glob.glob(os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data", "*_NDVI.tif"))
    et_files = glob.glob(os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}", "ET_PT_SINRH_ERA5", "*_PT_SINRH_ET.tif"))
    
    ndvi_dates = [os.path.basename(f).split('_')[0] for f in ndvi_files]
    et_dates = [os.path.basename(f).split('_')[0] for f in et_files]
    
    common_dates = sorted(list(set(ndvi_dates) & set(et_dates)))
    if not common_dates:
        print(f"Aucune date commune trouvée pour NDVI et ET sur le site {site}.")
        return None, None
        
    date_str = common_dates[-1] # Prendre la date la plus récente commune
    ndvi_path = [f for f in ndvi_files if date_str in f][0]
    
    print(f"Utilisation de la date : {date_str}")
    
    # Il faut d'abord calculer r_icos, c_icos
    with rasterio.open(ndvi_path) as src:
        transform = src.transform
        crs = src.crs
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x_icos, y_icos = transformer.transform(lon, lat)
        c_icos = int((x_icos - transform.c) / transform.a)
        r_icos = int((y_icos - transform.f) / transform.e)
        
    # Carte Pixels Entrainement
    pts, pts_crs = get_homogeneous_points(ndvi_path, r_icos, c_icos, num_points=15)
    
    with rasterio.open(ndvi_path) as src:
        ndvi_data = src.read(1)
        transform = src.transform
        crs = src.crs
        
        plt.figure(figsize=(12, 10))
        im = plt.imshow(ndvi_data, cmap='RdYlGn', vmin=-0.1, vmax=1.0)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="NDVI (Végétation)")
        
        # Pixel ICOS central en bleu a déjà été calculé
        
        # Autres pixels en rouge (pts contient des tuples (x_p, y_p) projetés)
        for i, (x_p, y_p) in enumerate(pts):
            c_p = int((x_p - transform.c) / transform.a)
            r_p = int((y_p - transform.f) / transform.e)
            
            # Ne pas redessiner le central
            if c_p == c_icos and r_p == r_icos:
                continue
                
            rect = patches.Rectangle((c_p-0.5, r_p-0.5), 1, 1, linewidth=1.5, edgecolor='magenta', facecolor='none')
            plt.gca().add_patch(rect)
            
        # Dessiner ICOS par dessus pour qu'il soit visible
        rect = patches.Rectangle((c_icos-0.5, r_icos-0.5), 1, 1, linewidth=3, edgecolor='blue', facecolor='none', label='Tour ICOS (Centre)')
        plt.gca().add_patch(rect)
            
        # Zoom autour d'ICOS
        window_size = 60
        if 0 <= r_icos - window_size and r_icos + window_size < ndvi_data.shape[0] and \
           0 <= c_icos - window_size and c_icos + window_size < ndvi_data.shape[1]:
            plt.xlim(c_icos - window_size, c_icos + window_size)
            plt.ylim(r_icos + window_size, r_icos - window_size)
            
        plt.title(f"Sélection des Pixels d'Entraînement - {site} ({date_str})", fontsize=14, pad=15)
        plt.legend(handles=[
            patches.Patch(edgecolor='blue', facecolor='none', label='Tour ICOS (Centre)', linewidth=3),
            patches.Patch(edgecolor='magenta', facecolor='none', label='Pixels Sélectionnés (Homogènes)', linewidth=1.5)
        ], loc='upper right')
        
        out_pix = os.path.join(out_dir_pix, f"{site}_{date_str}_Training_Pixels.png")
        plt.savefig(out_pix, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Carte pixels sauvegardée : {out_pix}")
        
    # 2. Carte ET PT-SINRH
    et_files = glob.glob(os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}", "ET_PT_SINRH_ERA5", f"{date_str}*_PT_SINRH_ET.tif"))
    if et_files:
        et_path = et_files[-1]
        with rasterio.open(et_path) as src:
            et_data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # Masquer les nans et valeurs aberrantes
            et_data = np.where((et_data < -9000) | (et_data > 100), np.nan, et_data)
            
            plt.figure(figsize=(12, 10))
            # Utiliser une colormap agréable pour l'eau (bleu/vert)
            im = plt.imshow(et_data, cmap='YlGnBu')
            plt.colorbar(im, fraction=0.046, pad=0.04, label="Évapotranspiration PT-SINRH (mm/h)")
            
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            x_icos, y_icos = transformer.transform(lon, lat)
            c_icos = int((x_icos - transform.c) / transform.a)
            r_icos = int((y_icos - transform.f) / transform.e)
            
            # Cadre large pour repérer ICOS facilement sur la carte ET
            rect = patches.Rectangle((c_icos-2, r_icos-2), 4, 4, linewidth=2.5, edgecolor='red', facecolor='none', label='Zone ICOS')
            plt.gca().add_patch(rect)
            
            window_size_et = 150 # Vue plus large pour l'ET (4.5 km de demi-côté)
            if 0 <= r_icos - window_size_et and r_icos + window_size_et < et_data.shape[0] and \
               0 <= c_icos - window_size_et and c_icos + window_size_et < et_data.shape[1]:
                plt.xlim(c_icos - window_size_et, c_icos + window_size_et)
                plt.ylim(r_icos + window_size_et, r_icos - window_size_et)
            
            plt.title(f"Carte Évapotranspiration PT-SINRH (Modèle Physique) - {site} ({date_str})", fontsize=14, pad=15)
            plt.legend(handles=[
                patches.Patch(edgecolor='red', facecolor='none', label='Zone ICOS', linewidth=2.5)
            ], loc='upper right')
            
            out_et = os.path.join(out_dir_et, f"{site}_{date_str}_ET_Map.png")
            plt.savefig(out_et, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Carte ET sauvegardée : {out_et}")
            
            return out_pix, out_et
    else:
        print(f"Fichier ET PT-SINRH non trouvé pour la date {date_str}")
        return out_pix, None

if __name__ == "__main__":
    plot_spatial_maps("Gebesee")
