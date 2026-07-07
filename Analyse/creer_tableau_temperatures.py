import os
import pandas as pd
import numpy as np

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Outputs")
CSV_PURE = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons", "Comparaison_ET_Landsat_vs_PureICOS.csv")
CSV_B10 = os.path.join(OUTPUTS_DIR, "Resultats_ET_TTME_ERA5_B10.csv")
OUT_CSV = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons", "Tableau_Temperatures_LST.csv")

def main():
    if not os.path.exists(CSV_PURE):
        print(f"❌ Fichier manquant : {CSV_PURE}")
        return
        
    # Charger DMS et ICOS
    df_pure = pd.read_csv(CSV_PURE)
    df_temp = df_pure[['Site', 'Date', 'LST_pixel (°C)', 'LST_ICOS (°C)']].copy()
    df_temp = df_temp.rename(columns={'LST_pixel (°C)': 'LST_Landsat_DMS'})
    
    # Charger B10
    if os.path.exists(CSV_B10):
        df_b10 = pd.read_csv(CSV_B10)
        df_b10 = df_b10[['Site', 'Date', 'LST_pixel (°C)']].copy()
        df_b10 = df_b10.rename(columns={'LST_pixel (°C)': 'LST_Landsat_B10'})
        
        # Fusionner
        df_temp = pd.merge(df_temp, df_b10, on=['Site', 'Date'], how='left')
    else:
        df_temp['LST_Landsat_B10'] = np.nan
        
    # Calculer les différences
    df_temp['Diff_DMS_vs_ICOS (°C)'] = df_temp['LST_Landsat_DMS'] - df_temp['LST_ICOS (°C)']
    df_temp['Diff_B10_vs_ICOS (°C)'] = df_temp['LST_Landsat_B10'] - df_temp['LST_ICOS (°C)']
    df_temp['Diff_DMS_vs_B10 (°C)'] = df_temp['LST_Landsat_DMS'] - df_temp['LST_Landsat_B10']
    
    # Arrondir
    for col in ['LST_Landsat_DMS', 'LST_Landsat_B10', 'LST_ICOS (°C)', 'Diff_DMS_vs_ICOS (°C)', 'Diff_B10_vs_ICOS (°C)', 'Diff_DMS_vs_B10 (°C)']:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].round(2)
            
    # Sauvegarder
    df_temp.to_csv(OUT_CSV, index=False)
    print(f"✅ Tableau des températures sauvegardé : {OUT_CSV}")
    
    # Statistiques par site
    print("\n📊 Biais Moyen de Température par Site (LST Satellite - LST ICOS) :")
    sites = df_temp['Site'].unique()
    for site in sites:
        df_site = df_temp[df_temp['Site'] == site]
        biais_dms = df_site['Diff_DMS_vs_ICOS (°C)'].mean()
        biais_b10 = df_site['Diff_B10_vs_ICOS (°C)'].mean()
        n = len(df_site)
        print(f"   {site:15s} (N={n:2d}) | Biais DMS = {biais_dms:>6.2f} °C | Biais B10 = {biais_b10:>6.2f} °C")
        
if __name__ == "__main__":
    import sys
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
