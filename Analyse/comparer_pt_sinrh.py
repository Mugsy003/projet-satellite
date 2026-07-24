"""
Comparaison PT-SINRH : ERA5 brut + Landsat vs référence ICOS + Landsat.

Ce script :
1. Pour chaque Site, construit la série temporelle complète des entrées
2. Calcule l'ET PT-SINRH sur toute la série
3. Génère des scatter plots et des séries temporelles.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyproj import Transformer
from sklearn.metrics import mean_squared_error

from Traitement.calcul_ET_PT_SINRH import calculate_pt_sinrh_et
from Traitement.calcul_ET import load_meteo_era5
from config import SITES_PILOTES as SITES

# ===========================================================================
# CONFIGURATION
# ===========================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "Outputs")
ICOS_METEO_DIR = os.path.join(PROJECT_ROOT, "Outputs_ICOS")
COMPARE_DIR = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons", "PT_SINRH")
os.makedirs(COMPARE_DIR, exist_ok=True)

FILE_ICOS = os.path.join(OUTPUTS_DIR, "Resultats_ET_TTME_ICOS.csv")
LAMBDA_V = 2.45e6  # Chaleur latente de vaporisation (J/kg)


# ===========================================================================
# FONCTIONS UTILITAIRES
# ===========================================================================

def _get_satellite_time(site, date_str):
    tif_folder = os.path.join(OUTPUTS_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data")
    default_dt = pd.to_datetime(f"{date_str} 10:30:00")
    if not os.path.exists(tif_folder):
        return default_dt
    tifs = glob.glob(os.path.join(tif_folder, f"{date_str}_*_{site}_NDVI.tif"))
    if not tifs:
        return default_dt
    basename = os.path.basename(tifs[0])
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})", basename)
    if match:
        date_part, hour, minute = match.groups()
        return pd.to_datetime(f"{date_part} {hour}:{minute}:00")
    return default_dt

def _get_pixel_value_from_tif(tif_path, lon, lat):
    try:
        with rasterio.open(tif_path) as src:
            transformer = Transformer.from_crs("EPSG:4326", str(src.crs), always_xy=True)
            x_p, y_p = transformer.transform(lon, lat)
            col_idx = int((x_p - src.transform.c) / src.transform.a)
            row_idx = int((y_p - src.transform.f) / src.transform.e)
            data = src.read(1)
            h, w = data.shape
            if 0 <= row_idx < h and 0 <= col_idx < w:
                val = data[row_idx, col_idx]
                if np.isfinite(val):
                    return float(val)
    except Exception:
        pass
    return np.nan

def _get_icos_meteo(site, target_dt):
    meteo_path = os.path.join(ICOS_METEO_DIR, f"donnees_icos_{site}.csv")
    if not os.path.exists(meteo_path):
        return None
    df = pd.read_csv(meteo_path)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    diff = abs(df['TIMESTAMP'] - target_dt)
    mask = diff <= pd.Timedelta(minutes=60)
    if df[mask].empty:
        return None
    idx_best = diff[mask].idxmin()
    row = df.loc[idx_best]
    return {
        'Ta':    row.get('TA_Consolide', np.nan),
        'u':     row.get('WS_Consolide', np.nan),
        'Rn':    row.get('Rn_Consolide', np.nan),
        'G':     row.get('G_Consolide', np.nan),
        'RH':    row.get('RH_Consolide', np.nan),
        'SW_IN': row.get('SW_IN_Consolide', np.nan),
        'VPD':   row.get('VPD_Consolide', np.nan),
    }

def _get_tmax_from_csv(csv_path, date_str, ta_col):
    if not os.path.exists(csv_path):
        return np.nan
    df = pd.read_csv(csv_path)
    first_col = df.columns[0]
    if first_col != 'TIMESTAMP':
        df.rename(columns={first_col: 'TIMESTAMP'}, inplace=True)
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP']).dt.tz_localize(None)
    day_start = pd.to_datetime(f"{date_str} 06:00:00")
    day_end = pd.to_datetime(f"{date_str} 18:00:00")
    mask = (df['TIMESTAMP'] >= day_start) & (df['TIMESTAMP'] <= day_end)
    df_day = df[mask]
    if df_day.empty or ta_col not in df_day.columns:
        return np.nan
    return df_day[ta_col].max()

def _calc_delta(T):
    return 4098.0 * 0.6108 * np.exp(17.27 * T / (T + 237.3)) / ((T + 237.3) ** 2)

def _calc_vpd_from_rh_ta(rh_pct, ta_c):
    e_sat = 0.6108 * np.exp(17.27 * ta_c / (ta_c + 237.3))
    return e_sat * (1.0 - rh_pct / 100.0)

def _build_cube(array_1d):
    return np.array(array_1d)[:, np.newaxis, np.newaxis]

def _run_pt_sinrh_series(inputs_dict):
    rh = inputs_dict['RH'] / 100.0
    rh = np.where(rh > 1.0, 1.0, rh)
    rh = np.clip(rh, 0.0, 1.0)
    
    rn = inputs_dict['Rn']
    
    g = inputs_dict['G']
    mask_g = np.isnan(g)
    g[mask_g] = rn[mask_g] * 0.1
    
    vpd = inputs_dict['VPD']
    vpd = np.maximum(vpd, 0.01)
    
    sw_in = inputs_dict['SW_IN']
    mask_sw = np.isnan(sw_in) | (sw_in <= 0)
    sw_in[mask_sw] = np.maximum(rn[mask_sw] * 1.2, 50.0)
    par = sw_in * 0.48
    
    t_max = inputs_dict['T_max']
    delta = _calc_delta(t_max)
    
    res = calculate_pt_sinrh_et(
        RH=_build_cube(rh), Rn=_build_cube(rn), G=_build_cube(g),
        T_max=_build_cube(t_max), NDVI=_build_cube(inputs_dict['NDVI']), 
        SAVI=_build_cube(inputs_dict['SAVI']), Delta=_build_cube(delta), 
        VPD=_build_cube(vpd), PAR=_build_cube(par)
    )
    
    le = res['ET'][:, 0, 0]
    return le * 3600.0 / LAMBDA_V

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    if not os.path.exists(FILE_ICOS):
        print(f"❌ Fichier introuvable : {FILE_ICOS}")
        return
        
    df_base = pd.read_csv(FILE_ICOS)
    df_base = df_base[~((df_base['Site'] == 'Gebesee') & (df_base['Date'].str.startswith('2023')))]
            
    final_rows = []
    
    for site in df_base['Site'].unique():
        if site not in SITES: continue
        lon, lat = SITES[site]['lon'], SITES[site]['lat']
        
        df_site = df_base[df_base['Site'] == site].sort_values('Date')
        site_data = []
        
        for _, row in df_site.iterrows():
            date_str = row['Date']
            dt = _get_satellite_time(site, date_str)
            
            tif_folder = os.path.join(OUTPUTS_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data")
            ndvi_f = glob.glob(os.path.join(tif_folder, f"{date_str}_*_{site}_NDVI.tif"))
            savi_f = glob.glob(os.path.join(tif_folder, f"{date_str}_*_{site}_SAVI.tif"))
            if not ndvi_f or not savi_f: continue
            
            ndvi = _get_pixel_value_from_tif(ndvi_f[0], lon, lat)
            savi = _get_pixel_value_from_tif(savi_f[0], lon, lat)
            if pd.isna(ndvi) or pd.isna(savi): continue
            
            m_icos = _get_icos_meteo(site, dt)
            m_era5 = load_meteo_era5(site, dt)
            
            if m_icos and m_era5 and not pd.isna(m_icos.get('Rn')) and not pd.isna(m_era5.get('Rn')):
                site_data.append({
                    'Date': date_str, 'NDVI': ndvi, 'SAVI': savi,
                    'm_icos': m_icos, 'm_era5': m_era5,
                    'TTME_ICOS': row['ET_pixel (mm/h)']
                })
                
        if not site_data: continue
        
        from collections import defaultdict
        site_data_by_year = defaultdict(list)
        for d in site_data:
            site_data_by_year[d['Date'][:4]].append(d)
            
        for year, s_data in site_data_by_year.items():
            N = len(s_data)
            inputs_icos, inputs_era5 = [
                {'NDVI': np.zeros(N), 'SAVI': np.zeros(N), 'RH': np.zeros(N), 'Rn': np.zeros(N),
                 'G': np.zeros(N), 'SW_IN': np.zeros(N), 'VPD': np.zeros(N), 'T_max': np.zeros(N)}
                for _ in range(2)
            ]
            
            for i, d in enumerate(s_data):
                for inp in [inputs_icos, inputs_era5]:
                inp['NDVI'][i] = d['NDVI']
                inp['SAVI'][i] = d['SAVI']
                
            m_i = d['m_icos']
            inputs_icos['RH'][i] = m_i['RH']
            inputs_icos['Rn'][i] = m_i['Rn']
            inputs_icos['G'][i] = m_i.get('G', np.nan)
            inputs_icos['SW_IN'][i] = m_i.get('SW_IN', np.nan)
            vpd = m_i.get('VPD', np.nan)
            if pd.isna(vpd): vpd = _calc_vpd_from_rh_ta(m_i['RH'], m_i['Ta'])
            elif vpd > 10: vpd /= 10.0
            inputs_icos['VPD'][i] = vpd
            tmax_i = _get_tmax_from_csv(os.path.join(ICOS_METEO_DIR, f"donnees_icos_{site}.csv"), d['Date'], 'TA_Consolide')
            inputs_icos['T_max'][i] = tmax_i if not pd.isna(tmax_i) else m_i['Ta'] + 3.0
            
            m_e = d['m_era5']
            inputs_era5['RH'][i] = m_e['RH']
            inputs_era5['Rn'][i] = m_e['Rn']
            inputs_era5['G'][i] = np.nan
            inputs_era5['SW_IN'][i] = m_e.get('R_s_down', np.nan)
                inputs_era5['VPD'][i] = _calc_vpd_from_rh_ta(m_e['RH'], m_e['Ta'])
                tmax_e = _get_tmax_from_csv(os.path.join(PROJECT_ROOT, "Outputs_ERA5", f"donnees_era5_{site}.csv"), d['Date'], 'Ta (°C)')
                inputs_era5['T_max'][i] = tmax_e if not pd.isna(tmax_e) else m_e['Ta'] + 3.0
                
            et_icos_arr = _run_pt_sinrh_series(inputs_icos)
            et_era5_arr = _run_pt_sinrh_series(inputs_era5)
            
            for i, d in enumerate(s_data):
                final_rows.append({
                    'Site': site,
                    'Date': d['Date'],
                    'NDVI': d['NDVI'],
                    'SAVI': d['SAVI'],
                    'ET_PT_ICOS (mm/h)': et_icos_arr[i],
                    'ET_PT_ERA5 (mm/h)': et_era5_arr[i],
                    'ET_TTME_ICOS (mm/h)': d['TTME_ICOS']
                })

    df_result = pd.DataFrame(final_rows)
    for col in ['ET_PT_ICOS (mm/h)', 'ET_PT_ERA5 (mm/h)']:
        df_result.loc[(df_result[col] > 5) | (df_result[col] < 0), col] = np.nan
        
    out_csv = os.path.join(COMPARE_DIR, "Comparaison_PT_SINRH.csv")
    df_result.to_csv(out_csv, index=False, encoding='utf-8-sig')

    ref_col = 'ET_PT_ICOS (mm/h)'
    for model_name, col in [("ERA5 brut", "ET_PT_ERA5 (mm/h)")]:
        df_valid = df_result.dropna(subset=[ref_col, col])
        if len(df_valid) >= 2:
            r = np.corrcoef(df_valid[ref_col], df_valid[col])[0, 1]
            rmse = np.sqrt(mean_squared_error(df_valid[ref_col], df_valid[col]))
            bias = np.mean(df_valid[col] - df_valid[ref_col])
            print(f"--- PT-SINRH {model_name} vs ICOS (N={len(df_valid)}) ---")
            print(f"   R² = {r**2:.4f} | RMSE = {rmse:.4f} mm/h | Biais = {bias:.4f} mm/h")
    
    # Graphiques globaux
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    df_v = df_result.dropna(subset=[ref_col, "ET_PT_ERA5 (mm/h)"])
    if len(df_v) >= 2:
        et_ref, et_mod = df_v[ref_col], df_v["ET_PT_ERA5 (mm/h)"]
        r = np.corrcoef(et_ref, et_mod)[0, 1]
        rmse = np.sqrt(mean_squared_error(et_ref, et_mod))
        bias = np.mean(et_mod - et_ref)
        ax.scatter(et_ref, et_mod, color="dodgerblue", alpha=0.7, edgecolors='w', s=50)
        min_v, max_v = min(et_ref.min(), et_mod.min()), max(et_ref.max(), et_mod.max())
        margin = (max_v - min_v) * 0.1 if max_v != min_v else 0.1
        ax.plot([min_v - margin, max_v + margin], [min_v - margin, max_v + margin], 'r--', linewidth=1.5)
        ax.set_title("PT-SINRH : ERA5 brut vs ICOS", fontsize=13, weight='bold')
        ax.set_xlabel("ET PT-SINRH ICOS (mm/h)")
        ax.set_ylabel("ET PT-SINRH ERA5 brut (mm/h)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.text(0.05, 0.95, f"N = {len(df_v)}\nR² = {r**2:.3f}\nRMSE = {rmse:.3f}\nBiais = {bias:.3f}", 
                transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.suptitle("PT-SINRH : Impact de la source météo", fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARE_DIR, "Scatter_PT_SINRH_ERA5_vs_ICOS.png"), dpi=150)
    plt.close()

    # Graphiques par site
    site_metrics = []
    for site in sorted(df_result['Site'].unique()):
        df_site = df_result[df_result['Site'] == site].sort_values('Date')
        if len(df_site) < 2: continue
        
        m = df_site.dropna(subset=[ref_col, "ET_PT_ERA5 (mm/h)"])
        if len(m) >= 2:
            r = np.corrcoef(m[ref_col], m["ET_PT_ERA5 (mm/h)"])[0, 1]
            rmse = np.sqrt(mean_squared_error(m[ref_col], m["ET_PT_ERA5 (mm/h)"]))
            bias = np.mean(m["ET_PT_ERA5 (mm/h)"] - m[ref_col])
            mae = np.mean(np.abs(m["ET_PT_ERA5 (mm/h)"] - m[ref_col]))
            r2_s, rmse_s, bias_s = f"{r**2:.3f}", f"{rmse:.3f}", f"{bias:.3f}"
            site_metrics.append({'Site': site, 'Modèle': 'ERA5 brut', 'r²': r**2, 'RMSE': rmse, 'MAE': mae})
        else:
            r2_s, rmse_s, bias_s = "N/A", "N/A", "N/A"
            
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        ax1 = axes[0]
        ax1.axis('tight'); ax1.axis('off')
        table = ax1.table(cellText=[["PT-SINRH ERA5 brut", r2_s, rmse_s, bias_s]], 
                          colLabels=["Modèle", "r²", "RMSE (mm/h)", "Biais (mm/h)"], loc='center')
        table.scale(1, 2)
        table.set_fontsize(11)
        for (r_idx, c_idx), cell in table.get_celld().items():
            if r_idx == 0: cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#2d6a4f')
        ax1.set_title("Performances vs PT-SINRH ICOS", fontsize=13, weight='bold', pad=20)
        
        ax2 = axes[1]
        valid = df_site.dropna(subset=[ref_col]).copy()
        valid['Date_obj'] = pd.to_datetime(valid['Date'])
        ax2.plot(valid['Date_obj'], valid[ref_col], marker='D', linestyle='--', color='purple', label='ICOS (réf.)', lw=2)
        valid_e = df_site.dropna(subset=['ET_PT_ERA5 (mm/h)']).copy()
        valid_e['Date_obj'] = pd.to_datetime(valid_e['Date'])
        ax2.plot(valid_e['Date_obj'], valid_e['ET_PT_ERA5 (mm/h)'], marker='s', linestyle='-', color='dodgerblue', label='ERA5 brut', lw=1.5)
        
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Évapotranspiration (mm/h)")
        ax2.set_title("Évolution Temporelle")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.suptitle(f"PT-SINRH — Site : {site}", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARE_DIR, f"Comparaison_PT_SINRH_{site}.png"), dpi=150)
        plt.close()

    if site_metrics:
        df_m = pd.DataFrame(site_metrics)
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        sites_sorted = sorted(df_m['Site'].unique())
        x = np.arange(len(sites_sorted))
        width = 0.5
        for ax, metric in zip(axes, ['RMSE', 'r²']):
            vals = [df_m[df_m['Site'] == s][metric].values[0] if len(df_m[df_m['Site'] == s]) > 0 else 0 for s in sites_sorted]
            bars = ax.bar(x, vals, width, color='dodgerblue', alpha=0.85)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.3f}", ha='center', fontsize=9)
            ax.set_xticks(x); ax.set_xticklabels(sites_sorted, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} par site (vs ICOS)", fontsize=13, weight='bold')
            ax.grid(True, axis='y', linestyle=':', alpha=0.5)
        plt.suptitle("PT-SINRH : Performances par site", fontsize=15, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARE_DIR, "Performances_PT_SINRH_par_Site.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
