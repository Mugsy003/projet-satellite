"""
Analyse/comparer_pt_sinrh.py
============================
Analyse des résultats de la méthode PT-SINRH de manière totalement
indépendante de TTME.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.dates as mdates
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from config import OUTPUT_DIR, SITES_PILOTES

COMPARE_DIR = os.path.join(OUTPUT_DIR, "Comparaisons")
os.makedirs(COMPARE_DIR, exist_ok=True)

def main():
    print("\n--- ANALYSE PT-SINRH ---")
    
    # Chemins des résultats
    path_era5 = os.path.join(OUTPUT_DIR, "Resultats_ET_PT_SINRH_ERA5.csv")
    path_icos = os.path.join(OUTPUT_DIR, "Resultats_ET_PT_SINRH_ICOS.csv")
    
    if not os.path.exists(path_era5):
        print(f"⚠️ Fichier manquant : {path_era5}")
        return
        
    df_era5 = pd.read_csv(path_era5)
    
    # Si on n'a pas encore calculé ICOS on essaie quand même
    if os.path.exists(path_icos):
        df_icos = pd.read_csv(path_icos)
    else:
        print("⚠️ Fichier ICOS manquant, on fera sans.")
        df_icos = pd.DataFrame(columns=['Site', 'Date', 'ET_PT_SINRH (mm/h)'])

    # Renommer les colonnes pour la fusion
    df_era5 = df_era5.rename(columns={'ET_PT_SINRH (mm/h)': 'ET_PT_ERA5 (mm/h)'})
    df_icos = df_icos.rename(columns={'ET_PT_SINRH (mm/h)': 'ET_PT_ICOS (mm/h)'})
    
    if 'Source_Meteo' in df_era5.columns: df_era5.drop(columns=['Source_Meteo'], inplace=True)
    if 'Source_Meteo' in df_icos.columns: df_icos.drop(columns=['Source_Meteo'], inplace=True)
    
    # Merge
    if not df_icos.empty:
        df_result = pd.merge(df_era5, df_icos, on=['Site', 'Date'], how='outer')
    else:
        df_result = df_era5
        df_result['ET_PT_ICOS (mm/h)'] = np.nan
        
    ref_col = 'ET_PT_ICOS (mm/h)'
    
    # Stats globales
    print(f"\nStats Globales :")
    m = df_result.dropna(subset=[ref_col, "ET_PT_ERA5 (mm/h)"])
    if len(m) >= 2:
        r = np.corrcoef(m[ref_col], m["ET_PT_ERA5 (mm/h)"])[0, 1]
        rmse = np.sqrt(mean_squared_error(m[ref_col], m["ET_PT_ERA5 (mm/h)"]))
        bias = np.mean(m["ET_PT_ERA5 (mm/h)"] - m[ref_col])
        print(f"   R² = {r**2:.4f} | RMSE = {rmse:.4f} mm/h | Biais = {bias:.4f} mm/h")
    else:
        print("   Pas assez de points pour calculer les stats globales.")

    # Graphique global
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    if len(m) >= 2:
        et_ref, et_mod = m[ref_col], m["ET_PT_ERA5 (mm/h)"]
        ax.scatter(et_ref, et_mod, color="dodgerblue", alpha=0.7, edgecolors='w', s=50)
        min_v, max_v = min(et_ref.min(), et_mod.min()), max(et_ref.max(), et_mod.max())
        margin = (max_v - min_v) * 0.1 if max_v != min_v else 0.1
        ax.plot([min_v - margin, max_v + margin], [min_v - margin, max_v + margin], 'r--', linewidth=1.5)
        ax.set_title("PT-SINRH : ERA5 vs ICOS", fontsize=13, weight='bold')
        ax.set_xlabel("ET PT-SINRH ICOS (mm/h)")
        ax.set_ylabel("ET PT-SINRH ERA5 (mm/h)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.text(0.05, 0.95, f"N = {len(m)}\nR² = {r**2:.3f}\nRMSE = {rmse:.3f}\nBiais = {bias:.3f}", 
                transform=ax.transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.suptitle("PT-SINRH : Validation globale", fontsize=15, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARE_DIR, "Scatter_PT_SINRH_ERA5_vs_ICOS.png"), dpi=150)
    plt.close()

    # Graphiques par site
    site_metrics = []
    for site in sorted(df_result['Site'].unique()):
        df_site = df_result[df_result['Site'] == site].sort_values('Date')
        if len(df_site) < 1: continue
        
        m_site = df_site.dropna(subset=[ref_col, "ET_PT_ERA5 (mm/h)"])
        if len(m_site) >= 2:
            r = np.corrcoef(m_site[ref_col], m_site["ET_PT_ERA5 (mm/h)"])[0, 1]
            rmse = np.sqrt(mean_squared_error(m_site[ref_col], m_site["ET_PT_ERA5 (mm/h)"]))
            bias = np.mean(m_site["ET_PT_ERA5 (mm/h)"] - m_site[ref_col])
            mae = np.mean(np.abs(m_site["ET_PT_ERA5 (mm/h)"] - m_site[ref_col]))
            r2_s, rmse_s, bias_s = f"{r**2:.3f}", f"{rmse:.3f}", f"{bias:.3f}"
            site_metrics.append({'Site': site, 'Modèle': 'ERA5', 'r²': r**2, 'RMSE': rmse, 'MAE': mae})
        else:
            r2_s, rmse_s, bias_s = "N/A", "N/A", "N/A"
            
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        ax1 = axes[0]
        ax1.axis('tight'); ax1.axis('off')
        table = ax1.table(cellText=[["PT-SINRH ERA5", r2_s, rmse_s, bias_s]], 
                          colLabels=["Modèle", "r²", "RMSE (mm/h)", "Biais (mm/h)"], loc='center')
        table.scale(1, 2)
        table.set_fontsize(11)
        for (r_idx, c_idx), cell in table.get_celld().items():
            if r_idx == 0: cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#2d6a4f')
        ax1.set_title("Performances vs PT-SINRH ICOS", fontsize=13, weight='bold', pad=20)
        
        ax2 = axes[1]
        valid = df_site.dropna(subset=[ref_col]).copy()
        if not valid.empty:
            valid['Date_obj'] = pd.to_datetime(valid['Date'])
            ax2.plot(valid['Date_obj'], valid[ref_col], marker='D', linestyle='--', color='purple', label='ICOS (réf.)', lw=2)
            
        valid_e = df_site.dropna(subset=['ET_PT_ERA5 (mm/h)']).copy()
        if not valid_e.empty:
            valid_e['Date_obj'] = pd.to_datetime(valid_e['Date'])
            ax2.plot(valid_e['Date_obj'], valid_e['ET_PT_ERA5 (mm/h)'], marker='s', linestyle='-', color='dodgerblue', label='ERA5', lw=1.5)
        
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Évapotranspiration (mm/h)")
        ax2.set_title("Évolution Temporelle")
        ax2.grid(True, linestyle=':', alpha=0.6)
        if not valid.empty or not valid_e.empty:
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
