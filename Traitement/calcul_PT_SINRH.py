"""
calcul_PT_SINRH.py
==================
Calcul de l'Évapotranspiration (ET) via le modèle PT-SINRH spatialisé.
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.stdout.reconfigure(encoding='utf-8')

from config import LOGGER, SITES_PILOTES, OUTPUT_DIR
from Traitement.calcul_ET_PT_SINRH import calculate_pt_sinrh_et
from Traitement.calcul_ET import load_meteo

LAMBDA_V = 2.45e6

def extract_datetime_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})(?:_(\d{2})h(\d{2}))?", filename)
    if match:
        date_str = match.group(1)
        hour_str = match.group(2) if match.group(2) else "10"
        min_str = match.group(3) if match.group(3) else "30"
        return pd.to_datetime(f"{date_str} {hour_str}:{min_str}:00")
    return None

def save_et_tif(data_array, transform, crs, output_path):
    if data_array is None: return
    h, w = data_array.shape
    try:
        with rasterio.open(
            output_path, 'w', driver='GTiff',
            height=h, width=w, count=1,
            dtype=np.float32, crs=crs,
            transform=transform, nodata=np.nan
        ) as dst:
            dst.write(data_array.astype(np.float32), 1)
    except Exception as e:
        LOGGER.error(f"Erreur d'écriture {output_path}: {e}")

def _calc_delta(T):
    return 4098.0 * 0.6108 * np.exp(17.27 * T / (T + 237.3)) / ((T + 237.3) ** 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, choices=['era5', 'icos', 'era5_ds'], default='era5')
    args = parser.parse_args()
    
    source = args.source
    BASE_TIF_DIR = OUTPUT_DIR
    resultats = []
    
    for site, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n{'='*60}")
        LOGGER.info(f"🌍 SITE (PT-SINRH) : {site}")
        
        tif_folder = os.path.join(BASE_TIF_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data")
        if not os.path.exists(tif_folder): continue
            
        output_dir = os.path.join(BASE_TIF_DIR, f"Serie_Temporelle_{site}", f"ET_PT_SINRH_{source.upper()}")
        os.makedirs(output_dir, exist_ok=True)
        
        dict_dates = {}
        for f in os.listdir(tif_folder):
            if not f.endswith('.tif'): continue
            dt = extract_datetime_from_filename(f)
            if not dt: continue
            if dt not in dict_dates: dict_dates[dt] = {}
            if "NDVI" in f: dict_dates[dt]['ndvi'] = os.path.join(tif_folder, f)
            if "SAVI" in f: dict_dates[dt]['savi'] = os.path.join(tif_folder, f)
        
        nb_ok = 0
        for target_dt, paths in sorted(dict_dates.items()):
            date_str = target_dt.strftime("%Y-%m-%d")
            path_ndvi = paths.get('ndvi')
            path_savi = paths.get('savi')
            if not path_ndvi or not path_savi: continue
                
            meteo = load_meteo(site, target_dt, date_str, source=source)
            if meteo is None or pd.isna(meteo.get('Rn')) or pd.isna(meteo.get('Ta')): continue
                
            try:
                with rasterio.open(path_ndvi) as src_ndvi:
                    ndvi_array = src_ndvi.read(1).astype(np.float32)
                    transform = src_ndvi.transform
                    crs = src_ndvi.crs
                with rasterio.open(path_savi) as src_savi:
                    savi_array = src_savi.read(1).astype(np.float32)
            except Exception as e:
                continue
                
            LOGGER.info(f"   🌿 {date_str} : Calcul PT-SINRH (Météo: {source.upper()})")
            
            h, w = ndvi_array.shape
            
            ta = meteo['Ta']
            rh = meteo.get('RH', 50.0)
            e_sat = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
            vpd = e_sat * (1.0 - rh / 100.0)
            sw_in = meteo.get('R_s_down', np.nan)
            if pd.isna(sw_in) or sw_in <= 0: sw_in = max(meteo['Rn']*1.2, 50.0)
            
            rh_arr = np.full((1, h, w), np.clip(rh/100.0, 0, 1), dtype=np.float32)
            rn_arr = np.full((1, h, w), meteo['Rn'], dtype=np.float32)
            g_arr = np.full((1, h, w), meteo.get('G', meteo['Rn']*0.1) if not pd.isna(meteo.get('G')) else meteo['Rn']*0.1, dtype=np.float32)
            t_max_arr = np.full((1, h, w), ta, dtype=np.float32) # Approximation
            vpd_arr = np.full((1, h, w), max(vpd, 0.01), dtype=np.float32)
            delta_arr = np.full((1, h, w), _calc_delta(ta), dtype=np.float32)
            par_arr = np.full((1, h, w), sw_in * 0.48, dtype=np.float32)
            
            ndvi_3d = ndvi_array[np.newaxis, :, :]
            savi_3d = savi_array[np.newaxis, :, :]
            
            res = calculate_pt_sinrh_et(
                RH=rh_arr, Rn=rn_arr, G=g_arr, T_max=t_max_arr, NDVI=ndvi_3d, SAVI=savi_3d,
                Delta=delta_arr, VPD=vpd_arr, PAR=par_arr
            )
            
            le_array = res['ET'][0, :, :]
            et_mm_h = le_array * 3600.0 / LAMBDA_V
            
            save_et_tif(et_mm_h, transform, crs, os.path.join(output_dir, f"{date_str}_{site}_PT_SINRH_ET.tif"))
            
            transformer_proj = Transformer.from_crs("EPSG:4326", str(crs), always_xy=True)
            x_p, y_p = transformer_proj.transform(coords["lon"], coords["lat"])
            col_idx = int((x_p - transform.c) / transform.a)
            row_idx = int((y_p - transform.f) / transform.e)
            
            if 0 <= row_idx < h and 0 <= col_idx < w:
                et_pixel = et_mm_h[row_idx, col_idx]
            else:
                et_pixel = np.nan
                
            LOGGER.info(f"      ✅ ET PT-SINRH carte moy = {np.nanmean(et_mm_h):.3f} mm/h | pixel = {et_pixel:.3f} mm/h")
            
            resultats.append({
                'Site': site,
                'Date': date_str,
                'ET_PT_SINRH (mm/h)': et_pixel
            })
            nb_ok += 1
            
    if resultats:
        df = pd.DataFrame(resultats)
        csv_path = os.path.join(OUTPUT_DIR, "Resultats_CSV", f"Resultats_ET_PT_SINRH_{source.upper()}.csv")
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            for site in df['Site'].unique():
                df_old = df_old[df_old['Site'] != site]
            df = pd.concat([df_old, df], ignore_index=True)
        df = df.sort_values(by=['Site', 'Date'])
        df.to_csv(csv_path, index=False)
        LOGGER.info(f"\n✅ Fichier PT-SINRH mis à jour : {csv_path}")

if __name__ == "__main__":
    main()