import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.dates as mdates
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR

COMPARE_DIR = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "2_Performances_PT_SINRH")
LAMBDA_V = 2.45e6

def get_icos_le(site, date_str):
    path = os.path.join("Outputs_ICOS", f"donnees_icos_LE_{site}.csv")
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path, index_col='TIMESTAMP', parse_dates=True)
    if 'LE_Consolide' not in df.columns:
        return np.nan
        
    # On cherche l'heure de passage autour de 10:30 UTC
    target_dt = pd.to_datetime(f"{date_str} 10:30:00")
    start = target_dt - pd.Timedelta(minutes=30)
    end = target_dt + pd.Timedelta(minutes=30)
    
    sub = df.loc[start:end]
    if sub.empty:
        return np.nan
        
    return sub['LE_Consolide'].mean()

def main():
    print("\n--- COMPARAISON PT-SINRH vs VRAI ICOS (LE) ---")
    
    path_era5 = os.path.join(OUTPUT_DIR, "Resultats_CSV", "Resultats_ET_PT_SINRH_ERA5.csv")
    path_ds = os.path.join(OUTPUT_DIR, "Resultats_CSV", "Resultats_ET_PT_SINRH_ERA5_DS.csv")
    
    if not os.path.exists(path_era5):
        print(f"Erreur Fichier manquant: {path_era5}")
        return

    df_era5 = pd.read_csv(path_era5).rename(columns={'ET_PT_SINRH (mm/h)': 'ET_ERA5 (mm/h)'})
    
    if os.path.exists(path_ds):
        df_ds = pd.read_csv(path_ds).rename(columns={'ET_PT_SINRH (mm/h)': 'ET_ERA5_DS (mm/h)'})
        df_era5 = pd.merge(df_era5, df_ds[['Site', 'Date', 'ET_ERA5_DS (mm/h)']], on=['Site', 'Date'], how='left')
    else:
        df_era5['ET_ERA5_DS (mm/h)'] = np.nan
        
    print("Extraction du LE ICOS pour chaque date...")
    
    # Récupérer LE
    df_era5['LE_ICOS (W/m2)'] = df_era5.apply(lambda row: get_icos_le(row['Site'], row['Date']), axis=1)
    df_era5['ET_Vrai_ICOS (mm/h)'] = df_era5['LE_ICOS (W/m2)'] * 3600.0 / LAMBDA_V
    
    # Sauvegarde CSV des résultats de comparaison
    df_era5.to_csv(os.path.join(OUTPUT_DIR, "Resultats_CSV", "Resultats_Comparaison_Vrai_ICOS.csv"), index=False)
    
    # Nettoyage pour les graphes
    df_valid = df_era5.dropna(subset=['ET_Vrai_ICOS (mm/h)']).copy()
    if df_valid.empty:
        print("Erreur: Aucune donnee valide en commun.")
        return
        
    print(f"{len(df_valid)} points de comparaison trouves.")
    
    site_metrics = []
    
    # Graphiques par site
    for site in sorted(df_valid['Site'].unique()):
        df_site = df_valid[df_valid['Site'] == site].sort_values('Date')
        df_site['Date_obj'] = pd.to_datetime(df_site['Date'])
        
        valid_era5 = df_site.dropna(subset=['ET_ERA5 (mm/h)', 'ET_Vrai_ICOS (mm/h)'])
        valid_ds = df_site.dropna(subset=['ET_ERA5_DS (mm/h)', 'ET_Vrai_ICOS (mm/h)'])
        
        if valid_era5.empty: continue
            
        r_era5 = np.corrcoef(valid_era5['ET_Vrai_ICOS (mm/h)'], valid_era5['ET_ERA5 (mm/h)'])[0, 1]
        rmse_era5 = np.sqrt(mean_squared_error(valid_era5['ET_Vrai_ICOS (mm/h)'], valid_era5['ET_ERA5 (mm/h)']))
        bias_era5 = np.mean(valid_era5['ET_ERA5 (mm/h)'] - valid_era5['ET_Vrai_ICOS (mm/h)'])
        
        r2_ds, rmse_ds, bias_ds = "N/A", "N/A", "N/A"
        if len(valid_ds) > 2:
            r = np.corrcoef(valid_ds['ET_Vrai_ICOS (mm/h)'], valid_ds['ET_ERA5_DS (mm/h)'])[0, 1]
            rmse = np.sqrt(mean_squared_error(valid_ds['ET_Vrai_ICOS (mm/h)'], valid_ds['ET_ERA5_DS (mm/h)']))
            bias = np.mean(valid_ds['ET_ERA5_DS (mm/h)'] - valid_ds['ET_Vrai_ICOS (mm/h)'])
            r2_ds, rmse_ds, bias_ds = f"{r**2:.3f}", f"{rmse:.3f}", f"{bias:.3f}"
            
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Table
        ax1 = axes[0]
        ax1.axis('tight'); ax1.axis('off')
        cellText = [
            ["ERA5", f"{r_era5**2:.3f}", f"{rmse_era5:.3f}", f"{bias_era5:.3f}"],
            ["ERA5_DS", r2_ds, rmse_ds, bias_ds]
        ]
        table = ax1.table(cellText=cellText, colLabels=["Modèle", "r²", "RMSE (mm/h)", "Biais (mm/h)"], loc='center')
        table.scale(1, 2); table.set_fontsize(11)
        for (r_idx, c_idx), cell in table.get_celld().items():
            if r_idx == 0: cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#d9534f')
        ax1.set_title("PT-SINRH vs VRAI ICOS (Tour à flux)", fontsize=13, weight='bold', pad=20)
        
        # Plot temporel
        ax2 = axes[1]
        ax2.plot(df_site['Date_obj'], df_site['ET_Vrai_ICOS (mm/h)'], marker='D', linestyle='--', color='purple', label='Vrai ICOS (Réf.)', lw=2)
        ax2.plot(valid_era5['Date_obj'], valid_era5['ET_ERA5 (mm/h)'], marker='s', linestyle='-', color='dodgerblue', label='PT-SINRH ERA5', lw=1.5)
        if not valid_ds.empty:
            ax2.plot(valid_ds['Date_obj'], valid_ds['ET_ERA5_DS (mm/h)'], marker='^', linestyle='-', color='darkorange', label='PT-SINRH ERA5-DS', lw=1.5)
            
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Évapotranspiration (mm/h)")
        ax2.set_title("Évolution Temporelle")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.suptitle(f"Vérité Terrain absolue (LE) — Site : {site}", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARE_DIR, f"Comparaison_Vrai_ICOS_{site}.png"), dpi=150)
        plt.close()
        
    print("\nAnalyse terminee. Les graphiques sont disponibles dans Outputs/Analyses_Graphiques/2_Performances_PT_SINRH.")

if __name__ == "__main__":
    main()
