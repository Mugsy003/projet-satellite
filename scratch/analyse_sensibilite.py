import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import mean_squared_error
import glob
import rasterio

# Importer les fonctions de l'algo TTME
sys.path.append(os.path.abspath("."))
from Traitement.calcul_ET import ttme_compute_et, calculer_bilan_radiatif, SITE_TTME_PARAMS, compute_fc

df_res_icos = pd.read_csv(r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Resultats_ET_TTME_ICOS.csv")
df_pure = pd.read_csv(r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Toutes_Comparaisons\Comparaison_ET_Landsat_vs_PureICOS.csv")

sites = df_res_icos['Site'].unique()
results = []

cache_file = "scratch/sensibilite_results_groupe.csv"

if os.path.exists(cache_file):
    print("Chargement des résultats existants...")
    df_res = pd.read_csv(cache_file)
else:
    print("Calcul des scénarios pour chaque date (Landsat vs ERA5 groupé)...")
    for site in sites:
        print(f"--- Traitement {site} ---")
        df_era5 = pd.read_csv(f"c:\\Users\\a951444\\Workspace\\projet-satellite\\Outputs_ERA5\\donnees_era5_{site}.csv")
        df_icos = pd.read_csv(f"c:\\Users\\a951444\\Workspace\\projet-satellite\\Outputs_ICOS\\donnees_icos_{site}.csv")
        
        df_era5['TIMESTAMP'] = pd.to_datetime(df_era5.iloc[:, 0]).dt.tz_localize(None)
        df_icos['TIMESTAMP'] = pd.to_datetime(df_icos['TIMESTAMP']).dt.tz_localize(None)
        
        dates = df_res_icos[df_res_icos['Site'] == site]['Date'].tolist()
        
        for date_str in dates:
            dt = pd.to_datetime(date_str)
            era5_dt = dt.replace(hour=10, minute=0, second=0)
            row_era5 = df_era5[df_era5['TIMESTAMP'] == era5_dt]
            
            mask_icos = (df_icos['TIMESTAMP'].dt.date == dt.date()) & (df_icos['TIMESTAMP'].dt.hour >= 9) & (df_icos['TIMESTAMP'].dt.hour <= 11)
            row_icos = df_icos[mask_icos]
            
            pure_row = df_pure[(df_pure['Site'] == site) & (df_pure['Date'] == date_str)]
            if row_era5.empty or row_icos.empty or pure_row.empty:
                continue
                
            row_icos = row_icos.iloc[0]
            row_era5 = row_era5.iloc[0]
            
            # Données ERA5
            Ta_col = [c for c in df_era5.columns if 'Ta (' in c or 'TA_Consolide' in c][0]
            Ta_era5 = row_era5[Ta_col]
            u_era5 = row_era5['u (m/s)']
            Rs_down_era5 = row_era5['R_s_down (W/m²)'] if 'R_s_down (W/m²)' in df_era5.columns else row_era5.get('SW_IN_Consolide', np.nan)
            Rl_down_era5 = row_era5['R_l_down (W/m²)'] if 'R_l_down (W/m²)' in df_era5.columns else row_era5.get('LW_IN_Consolide', np.nan)
            
            # Données ICOS
            Ta_icos = row_icos['TA_Consolide'] if 'TA_Consolide' in df_icos.columns else np.nan
            u_icos = row_icos['WS_Consolide'] if 'WS_Consolide' in df_icos.columns else 2.0
            Rn_icos = row_icos['Rn_Consolide'] if 'Rn_Consolide' in df_icos.columns else np.nan
            Rs_down_icos = row_icos['SW_IN_Consolide'] if 'SW_IN_Consolide' in df_icos.columns else np.nan
            Rl_down_icos = row_icos['LW_IN_Consolide'] if 'LW_IN_Consolide' in df_icos.columns else np.nan
            G_meas_icos = row_icos['G_Consolide'] if 'G_Consolide' in df_icos.columns else np.nan
            LST_icos = pure_row['LST_ICOS (°C)'].values[0]
            et_pure = pure_row['ET_pure_ICOS (mm/h)'].values[0]
            
            if pd.isna(Rs_down_icos) or pd.isna(Rl_down_icos) or pd.isna(LST_icos) or pd.isna(et_pure):
                continue
                
            base_dir = f"c:\\Users\\a951444\\Workspace\\projet-satellite\\Outputs\\Serie_Temporelle_{site}\\3_Indices\\TIF_Data"
            lst_files = glob.glob(f"{base_dir}\\{date_str}*LST_Sharpened_DMS.tif")
            ndvi_files = glob.glob(f"{base_dir}\\{date_str}*NDVI.tif")
            b2_files = glob.glob(f"{base_dir}\\{date_str}*B2.tif")
            
            if not lst_files or not ndvi_files or not b2_files:
                continue
                
            try:
                with rasterio.open(lst_files[0]) as src_lst:
                    lst_landsat_arr = src_lst.read(1).astype(np.float32)
                    lst_landsat_arr = np.where(lst_landsat_arr > 200, lst_landsat_arr - 273.15, lst_landsat_arr)
                    lst_landsat_arr = np.where((lst_landsat_arr < -50) | (lst_landsat_arr > 80), np.nan, lst_landsat_arr)
                    transform = src_lst.transform
                    crs = src_lst.crs
                
                with rasterio.open(ndvi_files[0]) as src_ndvi: ndvi_array = src_ndvi.read(1).astype(np.float32)
                with rasterio.open(b2_files[0]) as src_b2: b2 = src_b2.read(1).astype(np.float32)
                with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B4.tif")[0]) as src_b4: b4 = src_b4.read(1).astype(np.float32)
                with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B5.tif")[0]) as src_b5: b5 = src_b5.read(1).astype(np.float32)
                with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B6.tif")[0]) as src_b6: b6 = src_b6.read(1).astype(np.float32)
                with rasterio.open(glob.glob(f"{base_dir}\\{date_str}*B7.tif")[0]) as src_b7: b7 = src_b7.read(1).astype(np.float32)
                
                # Align shapes
                h_min = min(lst_landsat_arr.shape[0], ndvi_array.shape[0], b2.shape[0])
                w_min = min(lst_landsat_arr.shape[1], ndvi_array.shape[1], b2.shape[1])
                lst_landsat_arr = lst_landsat_arr[:h_min, :w_min]
                ndvi_array = ndvi_array[:h_min, :w_min]
                b2, b4, b5, b6, b7 = b2[:h_min, :w_min], b4[:h_min, :w_min], b5[:h_min, :w_min], b6[:h_min, :w_min], b7[:h_min, :w_min]
                
                row_station, col_station = int(h_min / 2), int(w_min / 2)
                
                alpha_array = (0.356 * b2 + 0.130 * b4 + 0.373 * b5 + 0.085 * b6 + 0.072 * b7 - 0.0018)
                alpha_array = np.clip(alpha_array, 0.01, 0.60)
                fc_array = compute_fc(ndvi_array)
                bandes = {'B2': b2, 'B4': b4, 'B5': b5, 'B6': b6, 'B7': b7}
                
                # Array où tout le monde a la LST de la station ICOS
                lst_icos_arr = np.full_like(lst_landsat_arr, LST_icos)
                
                def compute_scenario(Ta_val, u_val, Rs_val, Rl_val, lst_arr):
                    # Recalculer Rn_2d
                    Rn_2d = calculer_bilan_radiatif(bandes, fc_array, lst_arr, Rs_val, Rl_val)
                    res = ttme_compute_et(lst_arr, ndvi_array, alpha_array, Ta=Ta_val, u=u_val, Rn=Rn_2d, 
                                          R_s_down=Rs_val, R_l_down=Rl_val, params=SITE_TTME_PARAMS.get("default"),
                                          G_measured=G_meas_icos, transform=transform, crs=crs, z_m=2.0)
                    if res is not None:
                        return res['ET_mm_h'][row_station, col_station]
                    return np.nan
                    
                # 1. LST Landsat seul (reste = ICOS)
                et_lst = compute_scenario(Ta_icos, u_icos, Rs_down_icos, Rl_down_icos, lst_landsat_arr)
                
                # 2. Groupe ERA5 (Ta + u + Rn ERA5, LST = ICOS)
                et_era5_group = compute_scenario(Ta_era5, u_era5, Rs_down_era5, Rl_down_era5, lst_icos_arr)
                
                results.append({
                    'Site': site,
                    'Date': date_str,
                    'ET_pure_ICOS': et_pure,
                    'ET_LST_Landsat': et_lst,
                    'ET_Groupe_ERA5': et_era5_group
                })
                
            except Exception as e:
                continue

    df_res = pd.DataFrame(results)
    df_res.to_csv(cache_file, index=False)

# Analyse des erreurs
metrics = []
for site in df_res['Site'].unique():
    sub = df_res[df_res['Site'] == site]
    for col in ['ET_LST_Landsat', 'ET_Groupe_ERA5']:
        mask = sub['ET_pure_ICOS'].notna() & sub[col].notna()
        if mask.sum() >= 2:
            mse = mean_squared_error(sub.loc[mask, 'ET_pure_ICOS'], sub.loc[mask, col])
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(sub.loc[mask, 'ET_pure_ICOS'] - sub.loc[mask, col]))
            metrics.append({'Site': site, 'Variable': col.replace('ET_', ''), 'RMSE': rmse, 'MSE': mse, 'MAE': mae})

df_metrics = pd.DataFrame(metrics)
print(df_metrics)

df_pivot_mae = df_metrics.pivot(index='Site', columns='Variable', values='MAE')
df_pivot_mse = df_metrics.pivot(index='Site', columns='Variable', values='MSE')

colors = ['#1f77b4', '#ff7f0e'] # Bleu pour ERA5, Orange pour Landsat

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot MAE (Linéaire)
df_pivot_mae.plot(kind='bar', stacked=True, color=colors, ax=ax1, edgecolor='black', linewidth=0.5)
ax1.set_title("Sensibilité Linéaire : Climat ERA5 vs LST Landsat (MAE)", fontsize=14)
ax1.set_ylabel("Somme des MAE (mm/h)", fontsize=12)
ax1.legend(title='Source d\'Erreur', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.tick_params(axis='x', rotation=45)

totals_mae = df_pivot_mae.sum(axis=1).values
for i, c in enumerate(ax1.containers):
    labels = []
    for j, v in enumerate(c):
        val = v.get_height()
        if val > 0:
            pct = 100 * val / totals_mae[j]
            labels.append(f'{pct:.1f}%' if pct > 5 else '')
        else:
            labels.append('')
    ax1.bar_label(c, labels=labels, label_type='center', fontsize=10, color='black', weight='bold')

# Plot MSE (Variance, pour les pourcentages demandés)
df_pivot_mse.plot(kind='bar', stacked=True, color=colors, ax=ax2, edgecolor='black', linewidth=0.5)
ax2.set_title("Décomposition de la Variance : Climat ERA5 vs LST Landsat (MSE)", fontsize=14)
ax2.set_ylabel("Somme des Variances (MSE en (mm/h)²)", fontsize=12)
ax2.legend(title='Source d\'Erreur', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.tick_params(axis='x', rotation=45)

totals_mse = df_pivot_mse.sum(axis=1).values
for i, c in enumerate(ax2.containers):
    labels = []
    for j, v in enumerate(c):
        val = v.get_height()
        if val > 0:
            pct = 100 * val / totals_mse[j]
            labels.append(f'{pct:.1f}%' if pct > 5 else '')
        else:
            labels.append('')
    ax2.bar_label(c, labels=labels, label_type='center', fontsize=10, color='black', weight='bold')

plt.tight_layout()
plt.savefig("Outputs/Toutes_Comparaisons/Analyse_Sensibilite_Landsat_vs_ERA5.png", dpi=300, bbox_inches='tight')
print("Graphique sauvegardé dans Outputs/Toutes_Comparaisons/Analyse_Sensibilite_Landsat_vs_ERA5.png")
