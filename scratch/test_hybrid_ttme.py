import os
import glob
import pandas as pd
import numpy as np
import rasterio
from sklearn.metrics import r2_score
import sys

# Ajouter le workspace au path pour importer les fonctions
sys.path.append(r"c:\Users\a951444\Workspace\projet-satellite")
from Traitement.calcul_ET import ttme_compute_et, compute_fc, calculer_bilan_radiatif, SITE_TTME_PARAMS


# Lire les résultats existants pour connaître les dates validées
df_res_icos = pd.read_csv(r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Resultats_ET_TTME_ICOS.csv")
df_pure = pd.read_csv(r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Toutes_Comparaisons\Comparaison_ET_Landsat_vs_PureICOS.csv")

sites = df_res_icos['Site'].unique()
results = []

for site in sites:
    print(f"\n--- Traitement {site} ---")
    df_era5 = pd.read_csv(f"c:\\Users\\a951444\\Workspace\\projet-satellite\\Outputs_ERA5\\donnees_era5_{site}.csv")
    df_icos = pd.read_csv(f"c:\\Users\\a951444\\Workspace\\projet-satellite\\Outputs_ICOS\\donnees_icos_{site}.csv")
    
    df_era5['TIMESTAMP'] = pd.to_datetime(df_era5.iloc[:, 0]).dt.tz_localize(None)
    df_icos['TIMESTAMP'] = pd.to_datetime(df_icos['TIMESTAMP']).dt.tz_localize(None)
    
    dates = df_res_icos[df_res_icos['Site'] == site]['Date'].tolist()
    
    for date_str in dates:
        dt = pd.to_datetime(date_str)
        # Retrouver les lignes météo
        era5_dt = dt.replace(hour=10, minute=0, second=0)
        row_era5 = df_era5[df_era5['TIMESTAMP'] == era5_dt]
        
        mask_icos = (df_icos['TIMESTAMP'].dt.date == dt.date()) & (df_icos['TIMESTAMP'].dt.hour >= 9) & (df_icos['TIMESTAMP'].dt.hour <= 11)
        row_icos = df_icos[mask_icos]
        
        if row_era5.empty or row_icos.empty:
            continue
            
        row_icos = row_icos.iloc[0]
        # On utilise Ta de ERA5, mais tout le reste d'ICOS
        Ta_col = [c for c in df_era5.columns if 'Ta (' in c or 'TA_Consolide' in c][0]
        Ta_era5 = row_era5.iloc[0][Ta_col]
        
        u_icos = row_icos['WS_Consolide'] if 'WS_Consolide' in df_icos.columns else 2.0
        Rn_icos = row_icos['Rn_Consolide'] if 'Rn_Consolide' in df_icos.columns else np.nan
        R_s_down = row_icos['SW_IN_Consolide'] if 'SW_IN_Consolide' in df_icos.columns else np.nan
        R_l_down = row_icos['LW_IN_Consolide'] if 'LW_IN_Consolide' in df_icos.columns else np.nan
        G_meas = row_icos['G_Consolide'] if 'G_Consolide' in df_icos.columns else np.nan
        
        if pd.isna(R_s_down) or pd.isna(R_l_down) or pd.isna(u_icos):
            continue
            
        base_dir = f"c:\\Users\\a951444\\Workspace\\projet-satellite\\Outputs\\Serie_Temporelle_{site}\\3_Indices\\TIF_Data"
        lst_files = glob.glob(f"{base_dir}\\{date_str}*LST_Sharpened_DMS.tif")
        ndvi_files = glob.glob(f"{base_dir}\\{date_str}*NDVI.tif")
        b2_files = glob.glob(f"{base_dir}\\{date_str}*B2.tif")
        
        if not lst_files or not ndvi_files or not b2_files:
            continue
            
        try:
            with rasterio.open(lst_files[0]) as src_lst:
                lst_array = src_lst.read(1).astype(np.float32)
                lst_array = np.where(lst_array > 200, lst_array - 273.15, lst_array)
                lst_array = np.where((lst_array < -50) | (lst_array > 80), np.nan, lst_array)
                transform = src_lst.transform
                crs = src_lst.crs
            
            with rasterio.open(ndvi_files[0]) as src_ndvi: ndvi_array = src_ndvi.read(1).astype(np.float32)
            with rasterio.open(b2_files[0]) as src_b2: b2 = src_b2.read(1).astype(np.float32)
            with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B4.tif")[0]) as src_b4: b4 = src_b4.read(1).astype(np.float32)
            with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B5.tif")[0]) as src_b5: b5 = src_b5.read(1).astype(np.float32)
            with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B6.tif")[0]) as src_b6: b6 = src_b6.read(1).astype(np.float32)
            with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B7.tif")[0]) as src_b7: b7 = src_b7.read(1).astype(np.float32)
            
            # Align shapes
            h_min = min(lst_array.shape[0], ndvi_array.shape[0], b2.shape[0])
            w_min = min(lst_array.shape[1], ndvi_array.shape[1], b2.shape[1])
            lst_array = lst_array[:h_min, :w_min]
            ndvi_array = ndvi_array[:h_min, :w_min]
            b2, b4, b5, b6, b7 = b2[:h_min, :w_min], b4[:h_min, :w_min], b5[:h_min, :w_min], b6[:h_min, :w_min], b7[:h_min, :w_min]
            
            alpha_array = (0.356 * b2 + 0.130 * b4 + 0.373 * b5 + 0.085 * b6 + 0.072 * b7 - 0.0018)
            alpha_array = np.clip(alpha_array, 0.01, 0.60)
            
            fc_array = compute_fc(ndvi_array)
            bandes = {'B2': b2, 'B4': b4, 'B5': b5, 'B6': b6, 'B7': b7}
            Rn_2d = calculer_bilan_radiatif(bandes, fc_array, lst_array, R_s_down, R_l_down)
            
            # HYBRID TTME RUN (Ta=ERA5, u=ICOS, Rn=ICOS)
            res = ttme_compute_et(lst_array, ndvi_array, alpha_array, Ta=Ta_era5, u=u_icos, Rn=Rn_2d, 
                                  R_s_down=R_s_down, R_l_down=R_l_down, params=SITE_TTME_PARAMS.get("default"),
                                  G_measured=G_meas, transform=transform, crs=crs, z_m=2.0) # Using z_m=2.0 since wind is from ICOS!
            
            if res is not None:
                row_station, col_station = int(h_min / 2), int(w_min / 2)
                et_val = res['ET_mm_h'][row_station, col_station]
                
                et_orig_icos = df_res_icos[(df_res_icos['Site'] == site) & (df_res_icos['Date'] == date_str)]['ET_pixel (mm/h)'].values[0]
                pure_row = df_pure[(df_pure['Site'] == site) & (df_pure['Date'] == date_str)]
                et_pure = pure_row['ET_pure_ICOS (mm/h)'].values[0] if not pure_row.empty else np.nan
                
                results.append({
                    'Site': site,
                    'Date': date_str,
                    'ET_hybrid': et_val,
                    'ET_ICOS_model': et_orig_icos,
                    'ET_pure_ICOS': et_pure
                })
        except Exception as e:
            print(f"Error {site} {date_str}: {e}")
            import traceback
            traceback.print_exc()
            continue

df_res = pd.DataFrame(results)

print("\n\n==============================================================")
print(" R2 GLOBAL (Toutes stations confondues)")
print("==============================================================")
mask = df_res['ET_hybrid'].notna() & df_res['ET_ICOS_model'].notna()
print(f"Hybrid vs Modèle ICOS : {r2_score(df_res.loc[mask, 'ET_ICOS_model'], df_res.loc[mask, 'ET_hybrid']):.3f}")

mask2 = df_res['ET_hybrid'].notna() & df_res['ET_pure_ICOS'].notna()
print(f"Hybrid vs Pure ICOS   : {r2_score(df_res.loc[mask2, 'ET_pure_ICOS'], df_res.loc[mask2, 'ET_hybrid']):.3f}")

print("\n==============================================================")
print(" R2 PAR SITE : Hybride (Ta ERA5 + u/Rn ICOS) vs Vérité Terrain (Pure ICOS)")
print("==============================================================")
for site, grp in df_res.groupby('Site'):
    mask = grp['ET_hybrid'].notna() & grp['ET_pure_ICOS'].notna()
    if mask.sum() > 2:
        r2 = r2_score(grp.loc[mask, 'ET_pure_ICOS'], grp.loc[mask, 'ET_hybrid'])
        print(f"{site:15s}: R2 = {r2:6.3f} (N={mask.sum()})")
