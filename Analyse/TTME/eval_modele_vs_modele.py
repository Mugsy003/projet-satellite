import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

# Configuration des chemins
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Outputs")
COMPARE_DIR = os.path.join(OUTPUTS_DIR, "Analyses_Graphiques", "1_Performances_TTME")
os.makedirs(COMPARE_DIR, exist_ok=True)

FILE_ICOS = os.path.join(OUTPUTS_DIR, "Resultats_CSV", "Resultats_ET_TTME_ICOS.csv")
FILE_ERA5 = os.path.join(OUTPUTS_DIR, "Resultats_CSV", "Resultats_ET_TTME_ERA5.csv")
FILE_ERA5_DS = os.path.join(OUTPUTS_DIR, "Resultats_CSV", "Resultats_ET_TTME_ERA5_DS.csv")

def main():
    print("=" * 60)
    print("   COMPARAISON ET : ICOS vs ERA5 vs ERA5_DS")
    print("=" * 60)
    
    if not os.path.exists(FILE_ICOS):
        print(f"❌ Fichier introuvable : {FILE_ICOS}")
        sys.exit(1)
        
    if not os.path.exists(FILE_ERA5):
        print(f"❌ Fichier introuvable : {FILE_ERA5}")
        sys.exit(1)
        
    df_icos = pd.read_csv(FILE_ICOS)
    
    # Filter out Gebesee 2023 data as station values are wrong
    mask_gebesee_2023 = (df_icos['Site'] == 'Gebesee') & (df_icos['Date'].str.startswith('2023'))
    df_icos = df_icos[~mask_gebesee_2023]
    
    df_era5 = pd.read_csv(FILE_ERA5)
    
    # Charger ERA5_DS (optionnel)
    has_era5_ds = os.path.exists(FILE_ERA5_DS)
    if has_era5_ds:
        df_era5_ds = pd.read_csv(FILE_ERA5_DS)
        print(f"✅ Chargement ERA5_DS : {len(df_era5_ds)} calculs trouvés.")
    else:
        df_era5_ds = pd.DataFrame()
        print(f"⚠️ Fichier ERA5_DS introuvable, il sera ignoré.")
        
    print(f"✅ Chargement ICOS : {len(df_icos)} calculs trouvés.")
    print(f"✅ Chargement ERA5 : {len(df_era5)} calculs trouvés.")
    
    cols_to_keep = ['Site', 'Date', 'LE_pixel (W/m²)', 'ET_pixel (mm/h)', 'Rn (W/m²)', 'Ta (°C)', 'u (m/s)']
    df_icos = df_icos[[c for c in cols_to_keep if c in df_icos.columns]]
    df_era5 = df_era5[[c for c in cols_to_keep if c in df_era5.columns]]
    if has_era5_ds:
        df_era5_ds = df_era5_ds[[c for c in cols_to_keep if c in df_era5_ds.columns]]
    
    df_merged = pd.merge(df_icos, df_era5, on=['Site', 'Date'], suffixes=('_ICOS', '_ERA5'), how='outer')
    if has_era5_ds:
        df_merged = pd.merge(df_merged, df_era5_ds, on=['Site', 'Date'], how='outer')
        df_merged.rename(columns={
            'LE_pixel (W/m²)': 'LE_pixel (W/m²)_ERA5_DS',
            'ET_pixel (mm/h)': 'ET_pixel (mm/h)_ERA5_DS',
            'Ta (°C)': 'Ta (°C)_ERA5_DS',
        }, inplace=True)
    
    if len(df_merged) == 0:
        print("⚠️ Aucune date commune trouvée.")
        sys.exit(0)
        
    print(f"\n🔄 Fusion terminée : {len(df_merged)} dates communes pour comparaison.\n")
    
    var_et = 'ET_pixel (mm/h)'
    var_le = 'LE_pixel (W/m²)'
    
    df_merged_metrics = df_merged.dropna(subset=[f'{var_et}_ICOS', f'{var_et}_ERA5'])
    if len(df_merged_metrics) == 0:
        sys.exit(0)
        
    et_icos = df_merged_metrics[f'{var_et}_ICOS']
    et_era5 = df_merged_metrics[f'{var_et}_ERA5']
    rmse_et = np.sqrt(mean_squared_error(et_icos, et_era5))
    bias_et = np.mean(et_era5 - et_icos)
    r2_et = r2_score(et_icos, et_era5)
    
    le_icos = df_merged_metrics[f'{var_le}_ICOS']
    le_era5 = df_merged_metrics[f'{var_le}_ERA5']
    rmse_le = np.sqrt(mean_squared_error(le_icos, le_era5))
    bias_le = np.mean(le_era5 - le_icos)
    r2_le = r2_score(le_icos, le_era5)
    
    print("-" * 40)
    print("MÉTRIQUES (ERA5 vs ICOS)")
    print(f"ET: RMSE={rmse_et:.4f}, Biais={bias_et:.4f}, R²={r2_et:.4f}")
    print(f"LE: RMSE={rmse_le:.2f}, Biais={bias_le:.2f}, R²={r2_le:.2f}")
    
    if has_era5_ds and f'{var_et}_ERA5_DS' in df_merged_metrics.columns:
        et_ds = df_merged_metrics[f'{var_et}_ERA5_DS']
        mask = et_ds.notna() & et_icos.notna()
        if mask.sum() > 0:
            rmse_et_ds = np.sqrt(mean_squared_error(et_icos[mask], et_ds[mask]))
            bias_et_ds = np.mean(et_ds[mask] - et_icos[mask])
            r2_et_ds = r2_score(et_icos[mask], et_ds[mask])
            
            le_ds = df_merged_metrics[f'{var_le}_ERA5_DS']
            rmse_le_ds = np.sqrt(mean_squared_error(le_icos[mask], le_ds[mask]))
            bias_le_ds = np.mean(le_ds[mask] - le_icos[mask])
            r2_le_ds = r2_score(le_icos[mask], le_ds[mask])
            
            print("-" * 40)
            print("MÉTRIQUES (ERA5_DS vs ICOS)")
            print(f"ET: RMSE={rmse_et_ds:.4f}, Biais={bias_et_ds:.4f}, R²={r2_et_ds:.4f}")
            print(f"LE: RMSE={rmse_le_ds:.2f}, Biais={bias_le_ds:.2f}, R²={r2_le_ds:.2f}")
    
    # Sauvegarde CSV
    out_csv = os.path.join(COMPARE_DIR, "Comparaison_ET_ICOS_vs_ERA5.csv")
    df_merged.to_csv(out_csv, index=False)
    
    # Tracé
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot ET
    ax = axes[0]
    ax.scatter(et_icos, et_era5, color='blue', alpha=0.5, label=f'ERA5 brute (RMSE={rmse_et:.3f})')
    if has_era5_ds:
        ax.scatter(et_icos[mask], et_ds[mask], color='green', alpha=0.5, label=f'ERA5 Downscalée (RMSE={rmse_et_ds:.3f})')
    
    min_val = df_merged[[f'{var_et}_ICOS', f'{var_et}_ERA5']].min().min()
    max_val = df_merged[[f'{var_et}_ICOS', f'{var_et}_ERA5']].max().max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
    ax.set_title("Évapotranspiration (ET) en mm/h")
    ax.set_xlabel("ET ICOS")
    ax.set_ylabel("ET Modèle")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
            
    # Plot LE
    ax = axes[1]
    ax.scatter(le_icos, le_era5, color='blue', alpha=0.5, label=f'ERA5 brute (RMSE={rmse_le:.1f})')
    if has_era5_ds:
        ax.scatter(le_icos[mask], le_ds[mask], color='green', alpha=0.5, label=f'ERA5 Downscalée (RMSE={rmse_le_ds:.1f})')
        
    min_val = df_merged[[f'{var_le}_ICOS', f'{var_le}_ERA5']].min().min()
    max_val = df_merged[[f'{var_le}_ICOS', f'{var_le}_ERA5']].max().max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
    ax.set_title("Flux de Chaleur Latente (LE) en W/m²")
    ax.set_xlabel("LE ICOS")
    ax.set_ylabel("LE Modèle")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
            
    plt.suptitle("Comparaison du modèle TTME selon la source météo (ERA5 vs ICOS)", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_png = os.path.join(COMPARE_DIR, "Scatter_ET_ICOS_vs_ERA5.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Graphique sauvegardé dans :\n   {out_png}")

if __name__ == "__main__":
    main()
