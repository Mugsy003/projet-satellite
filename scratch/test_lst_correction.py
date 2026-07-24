import os
import sys
import pandas as pd
import numpy as np
import rasterio
import glob
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath("."))
from Traitement.calcul_ET import ttme_compute_et, calculer_bilan_radiatif, SITE_TTME_PARAMS, compute_fc

df_res_icos = pd.read_csv(r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Resultats_ET_TTME_ICOS.csv")
df_pure = pd.read_csv(r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Toutes_Comparaisons\Comparaison_ET_Landsat_vs_PureICOS.csv")
df_era5_ds = pd.read_csv(r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Resultats_ET_TTME_ERA5_DS.csv")

sites = df_res_icos['Site'].unique()
results = []

cache_file = "scratch/test_lst_correction_results.csv"

if os.path.exists(cache_file):
    print("Chargement des résultats existants...")
    df_res = pd.read_csv(cache_file)
else:
    print("Calcul des scénarios de correction LST pour chaque date...")
    for site in sites:
        print(f"--- Traitement {site} ---")
        df_era5_brut = pd.read_csv(f"c:\\Users\\a951444\\Workspace\\projet-satellite\\Outputs_ERA5\\donnees_era5_{site}.csv")
        df_era5_brut['TIMESTAMP'] = pd.to_datetime(df_era5_brut.iloc[:, 0]).dt.tz_localize(None)
        
        dates = df_res_icos[df_res_icos['Site'] == site]['Date'].tolist()
        
        for date_str in dates:
            dt = pd.to_datetime(date_str)
            era5_dt = dt.replace(hour=10, minute=0, second=0)
            
            row_era5_brut = df_era5_brut[df_era5_brut['TIMESTAMP'] == era5_dt]
            pure_row = df_pure[(df_pure['Site'] == site) & (df_pure['Date'] == date_str)]
            row_ds = df_era5_ds[(df_era5_ds['Site'] == site) & (df_era5_ds['Date'] == date_str)]
            
            if row_era5_brut.empty or pure_row.empty or row_ds.empty:
                continue
                
            row_era5_brut = row_era5_brut.iloc[0]
            et_pure = pure_row['ET_pure_ICOS (mm/h)'].values[0]
            if pd.isna(et_pure):
                continue
                
            # Extraire les météos
            Ta_ds = row_ds['Ta (°C)'].values[0]  # C'est la Ta downscalée
            u_era5 = row_era5_brut['u (m/s)']
            Rs_down_era5 = row_era5_brut['R_s_down (W/m²)'] if 'R_s_down (W/m²)' in df_era5_brut.columns else row_era5_brut.get('SW_IN_Consolide', np.nan)
            Rl_down_era5 = row_era5_brut['R_l_down (W/m²)'] if 'R_l_down (W/m²)' in df_era5_brut.columns else row_era5_brut.get('LW_IN_Consolide', np.nan)
            
            if pd.isna(Rs_down_era5) or pd.isna(Rl_down_era5):
                continue
            
            base_dir = f"c:\\Users\\a951444\\Workspace\\projet-satellite\\Outputs\\Serie_Temporelle_{site}\\3_Indices\\TIF_Data"
            lst_files = glob.glob(f"{base_dir}\\{date_str}*LST_Sharpened_DMS.tif")
            ndvi_files = glob.glob(f"{base_dir}\\{date_str}*NDVI.tif")
            b2_files = glob.glob(f"{base_dir}\\{date_str}*B2.tif")
            
            if not lst_files or not ndvi_files or not b2_files:
                continue
                
            try:
                with rasterio.open(lst_files[0]) as src_lst:
                    lst_base = src_lst.read(1).astype(np.float32)
                    lst_base = np.where(lst_base > 200, lst_base - 273.15, lst_base)
                    lst_base = np.where((lst_base < -50) | (lst_base > 80), np.nan, lst_base)
                    transform = src_lst.transform
                    crs = src_lst.crs
                
                with rasterio.open(ndvi_files[0]) as src_ndvi: ndvi_array = src_ndvi.read(1).astype(np.float32)
                with rasterio.open(b2_files[0]) as src_b2: b2 = src_b2.read(1).astype(np.float32)
                with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B4.tif")[0]) as src_b4: b4 = src_b4.read(1).astype(np.float32)
                with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B5.tif")[0]) as src_b5: b5 = src_b5.read(1).astype(np.float32)
                with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B6.tif")[0]) as src_b6: b6 = src_b6.read(1).astype(np.float32)
                with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B7.tif")[0]) as src_b7: b7 = src_b7.read(1).astype(np.float32)
                
                h_min = min(lst_base.shape[0], ndvi_array.shape[0], b2.shape[0])
                w_min = min(lst_base.shape[1], ndvi_array.shape[1], b2.shape[1])
                lst_base = lst_base[:h_min, :w_min]
                ndvi_array = ndvi_array[:h_min, :w_min]
                b2, b4, b5, b6, b7 = b2[:h_min, :w_min], b4[:h_min, :w_min], b5[:h_min, :w_min], b6[:h_min, :w_min], b7[:h_min, :w_min]
                
                row_station, col_station = int(h_min / 2), int(w_min / 2)
                
                alpha_array = (0.356 * b2 + 0.130 * b4 + 0.373 * b5 + 0.085 * b6 + 0.072 * b7 - 0.0018)
                alpha_array = np.clip(alpha_array, 0.01, 0.60)
                fc_array = compute_fc(ndvi_array)
                bandes = {'B2': b2, 'B4': b4, 'B5': b5, 'B6': b6, 'B7': b7}
                
                # CREATION DES SCENARIOS LST
                lst_095 = lst_base * 0.95
                lst_minus1 = np.where(lst_base > 25, lst_base - 1.0, lst_base)
                
                def compute_scenario(lst_arr):
                    Rn_2d = calculer_bilan_radiatif(bandes, fc_array, lst_arr, Rs_down_era5, Rl_down_era5)
                    res = ttme_compute_et(lst_arr, ndvi_array, alpha_array, Ta=Ta_ds, u=u_era5, Rn=Rn_2d, 
                                          R_s_down=Rs_down_era5, R_l_down=Rl_down_era5, params=SITE_TTME_PARAMS.get("default"),
                                          G_measured=None, transform=transform, crs=crs, z_m=2.0)
                    if res is not None:
                        return res['ET_mm_h'][row_station, col_station]
                    return np.nan
                    
                et_base = compute_scenario(lst_base)
                et_095 = compute_scenario(lst_095)
                et_minus1 = compute_scenario(lst_minus1)
                
                results.append({
                    'Site': site,
                    'Date': date_str,
                    'ET_pure_ICOS': et_pure,
                    'ET_Base': et_base,
                    'ET_095': et_095,
                    'ET_Minus1': et_minus1
                })
                
            except Exception as e:
                continue

    df_res = pd.DataFrame(results)
    df_res.to_csv(cache_file, index=False)

# Analyse des erreurs
print("\n--- RESULTATS DES CORRECTIONS LST (MAE) ---")
for col in ['ET_Base', 'ET_095', 'ET_Minus1']:
    mask = df_res['ET_pure_ICOS'].notna() & df_res[col].notna()
    mae = np.mean(np.abs(df_res.loc[mask, 'ET_pure_ICOS'] - df_res.loc[mask, col]))
    print(f"MAE Global ({col}) : {mae:.4f} mm/h")

print("\n--- MAE PAR SITE ---")
metrics = []
for site in df_res['Site'].unique():
    sub = df_res[df_res['Site'] == site]
    for col in ['ET_Base', 'ET_095', 'ET_Minus1']:
        mask = sub['ET_pure_ICOS'].notna() & sub[col].notna()
        if mask.sum() >= 2:
            mae = np.mean(np.abs(sub.loc[mask, 'ET_pure_ICOS'] - sub.loc[mask, col]))
            metrics.append({'Site': site, 'Scenario': col, 'MAE': mae})

df_metrics = pd.DataFrame(metrics).pivot(index='Site', columns='Scenario', values='MAE')
print(df_metrics)
