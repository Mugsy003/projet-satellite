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

COMPARE_DIR = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "2_Performances_PT_SINRH")
os.makedirs(COMPARE_DIR, exist_ok=True)

def main():
    print("\n--- ANALYSE PT-SINRH ---")
    
    path_era5 = os.path.join(OUTPUT_DIR, "Resultats_CSV", "Resultats_ET_PT_SINRH_ERA5.csv")
    path_icos = os.path.join(OUTPUT_DIR, "Resultats_CSV", "Resultats_ET_PT_SINRH_ICOS.csv")
    path_ds = os.path.join(OUTPUT_DIR, "Resultats_CSV", "Resultats_ET_PT_SINRH_ERA5_DS.csv")
    
    if not os.path.exists(path_era5):
        print(f"⚠️ Fichier manquant : {path_era5}")
        return
        
    df_era5 = pd.read_csv(path_era5).rename(columns={'ET_PT_SINRH (mm/h)': 'ET_PT_ERA5 (mm/h)'})
    
    if os.path.exists(path_icos):
        df_icos = pd.read_csv(path_icos).rename(columns={'ET_PT_SINRH (mm/h)': 'ET_PT_ICOS (mm/h)'})
    else:
        df_icos = pd.DataFrame(columns=['Site', 'Date', 'ET_PT_ICOS (mm/h)'])

    if os.path.exists(path_ds):
        df_ds = pd.read_csv(path_ds).rename(columns={'ET_PT_SINRH (mm/h)': 'ET_PT_ERA5_DS (mm/h)'})
    else:
        df_ds = pd.DataFrame(columns=['Site', 'Date', 'ET_PT_ERA5_DS (mm/h)'])

    for df in [df_era5, df_icos, df_ds]:
        if 'Source_Meteo' in df.columns: df.drop(columns=['Source_Meteo'], inplace=True)
    
    # Merge inner pour n'avoir que les jours où on a ICOS
    if not df_icos.empty:
        df_result = pd.merge(df_era5, df_icos, on=['Site', 'Date'], how='inner')
    else:
        df_result = df_era5
        df_result['ET_PT_ICOS (mm/h)'] = np.nan

    if not df_ds.empty:
        df_result = pd.merge(df_result, df_ds, on=['Site', 'Date'], how='left')
    else:
        df_result['ET_PT_ERA5_DS (mm/h)'] = np.nan
        
    ref_col = 'ET_PT_ICOS (mm/h)'
    
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

        valid_ds = df_site.dropna(subset=['ET_PT_ERA5_DS (mm/h)']).copy()
        if not valid_ds.empty:
            valid_ds['Date_obj'] = pd.to_datetime(valid_ds['Date'])
            ax2.plot(valid_ds['Date_obj'], valid_ds['ET_PT_ERA5_DS (mm/h)'], marker='^', linestyle='-', color='darkorange', label='ERA5-DS', lw=1.5)
        
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Évapotranspiration (mm/h)")
        ax2.set_title("Évolution Temporelle")
        ax2.grid(True, linestyle=':', alpha=0.6)
        if not valid.empty or not valid_e.empty or not valid_ds.empty:
            ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        plt.suptitle(f"PT-SINRH — Site : {site}", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARE_DIR, f"Comparaison_PT_SINRH_{site}.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
