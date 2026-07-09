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
COMPARE_DIR = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons")
os.makedirs(COMPARE_DIR, exist_ok=True)

FILE_ICOS = os.path.join(OUTPUTS_DIR, "Resultats_ET_TTME.csv")
FILE_ERA5 = os.path.join(OUTPUTS_DIR, "Resultats_ET_TTME_ERA5.csv")

def main():
    print("=" * 60)
    print("   COMPARAISON ET : ICOS vs ERA5")
    print("=" * 60)
    
    if not os.path.exists(FILE_ICOS):
        print(f"❌ Fichier introuvable : {FILE_ICOS}")
        print("   Veuillez lancer Traitement/calcul_ET.py avec la source ICOS d'abord.")
        sys.exit(1)
        
    if not os.path.exists(FILE_ERA5):
        print(f"❌ Fichier introuvable : {FILE_ERA5}")
        print("   Veuillez attendre la fin de l'extraction ERA5 et lancer Traitement/calcul_ET.py --source era5")
        sys.exit(1)
        
    df_icos = pd.read_csv(FILE_ICOS)
    df_era5 = pd.read_csv(FILE_ERA5)
    
    print(f"✅ Chargement ICOS : {len(df_icos)} calculs trouvés.")
    print(f"✅ Chargement ERA5 : {len(df_era5)} calculs trouvés.")
    
    # Renommer les colonnes d'intérêt pour la fusion
    cols_to_keep = ['Site', 'Date', 'LE_pixel (W/m²)', 'ET_pixel (mm/h)', 'Rn (W/m²)', 'Ta (°C)', 'u (m/s)']
    df_icos = df_icos[[c for c in cols_to_keep if c in df_icos.columns]]
    df_era5 = df_era5[[c for c in cols_to_keep if c in df_era5.columns]]
    
    # Suffixes pour différencier
    df_merged = pd.merge(df_icos, df_era5, on=['Site', 'Date'], suffixes=('_ICOS', '_ERA5'))
    
    if len(df_merged) == 0:
        print("⚠️ Aucune date commune trouvée entre les deux jeux de résultats.")
        sys.exit(0)
        
    print(f"\n🔄 Fusion terminée : {len(df_merged)} dates communes pour comparaison.\n")
    
    # Variables à comparer
    var_et = 'ET_pixel (mm/h)'
    var_le = 'LE_pixel (W/m²)'
    
    # Nettoyage des NaNs potentiels
    df_merged = df_merged.dropna(subset=[f'{var_et}_ICOS', f'{var_et}_ERA5'])
    
    if len(df_merged) == 0:
        print("⚠️ Aucune donnée valide (non-NaN) pour l'ET.")
        sys.exit(0)
        
    # Calcul des métriques pour l'ET (mm/h)
    et_icos = df_merged[f'{var_et}_ICOS']
    et_era5 = df_merged[f'{var_et}_ERA5']
    
    rmse_et = np.sqrt(mean_squared_error(et_icos, et_era5))
    bias_et = np.mean(et_era5 - et_icos)
    r2_et = r2_score(et_icos, et_era5)
    
    # Calcul des métriques pour LE (W/m²)
    le_icos = df_merged[f'{var_le}_ICOS']
    le_era5 = df_merged[f'{var_le}_ERA5']
    
    rmse_le = np.sqrt(mean_squared_error(le_icos, le_era5))
    bias_le = np.mean(le_era5 - le_icos)
    r2_le = r2_score(le_icos, le_era5)
    
    print("-" * 40)
    print("MÉTRIQUES DE COMPARAISON (ERA5 vs ICOS)")
    print("-" * 40)
    print(f"ET (mm/h) :")
    print(f"  RMSE = {rmse_et:.4f}")
    print(f"  Biais= {bias_et:.4f}")
    print(f"  R²   = {r2_et:.4f}")
    print(f"\nLE (W/m²) :")
    print(f"  RMSE = {rmse_le:.2f}")
    print(f"  Biais= {bias_le:.2f}")
    print(f"  R²   = {r2_le:.2f}")
    print("-" * 40)
    
    # Sauvegarde CSV
    out_csv = os.path.join(COMPARE_DIR, "Comparaison_ET_ICOS_vs_ERA5.csv")
    df_merged.to_csv(out_csv, index=False)
    print(f"\n💾 Détails de la comparaison sauvegardés dans :\n   {out_csv}")
    
    # Tracé de la figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot ET
    ax = axes[0]
    ax.scatter(et_icos, et_era5, color='blue', alpha=0.7)
    min_val = min(et_icos.min(), et_era5.min())
    max_val = max(et_icos.max(), et_era5.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
    ax.set_title("Évapotranspiration (ET) en mm/h")
    ax.set_xlabel("ET mesurée/calculée avec ICOS")
    ax.set_ylabel("ET calculée avec ERA5")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    ax.text(0.05, 0.95, f"R² = {r2_et:.2f}\nRMSE = {rmse_et:.3f}\nBiais = {bias_et:.3f}", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    # Plot LE
    ax = axes[1]
    ax.scatter(le_icos, le_era5, color='orange', alpha=0.7)
    min_val = min(le_icos.min(), le_era5.min())
    max_val = max(le_icos.max(), le_era5.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
    ax.set_title("Flux de Chaleur Latente (LE) en W/m²")
    ax.set_xlabel("LE ICOS (W/m²)")
    ax.set_ylabel("LE ERA5 (W/m²)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    ax.text(0.05, 0.95, f"R² = {r2_le:.2f}\nRMSE = {rmse_le:.1f}\nBiais = {bias_le:.1f}", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
    plt.suptitle("Comparaison du modèle TTME selon la source météo (ERA5 vs ICOS)", fontsize=14, y=1.02)
    plt.tight_layout()
    
    out_png = os.path.join(COMPARE_DIR, "Scatter_ET_ICOS_vs_ERA5.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Graphique sauvegardé dans :\n   {out_png}")

if __name__ == "__main__":
    main()
