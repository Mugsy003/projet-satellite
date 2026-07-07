import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
sys.stdout.reconfigure(encoding='utf-8')

# Chemin vers les résultats
RESULTATS_ERA5 = r"C:\Users\a951444\Workspace\projet-satellite\Outputs\Resultats_ET_TTME_ERA5.csv"
RESULTATS_ICOS = r"C:\Users\a951444\Workspace\projet-satellite\Outputs\Resultats_ET_TTME.csv"

def main():
    if not os.path.exists(RESULTATS_ERA5) or not os.path.exists(RESULTATS_ICOS):
        print(f"❌ Les fichiers de résultats sont introuvables.")
        return
        
    df_era5 = pd.read_csv(RESULTATS_ERA5)
    df_icos = pd.read_csv(RESULTATS_ICOS)
    
    # On renomme les colonnes d'intérêt pour éviter les conflits lors de la fusion
    col_et = 'ET_pixel (mm/h)'
    col_le = 'LE_pixel (W/m²)'
    
    df_era5 = df_era5[['Site', 'Date', col_et, col_le]].rename(columns={
        col_et: 'ET_ERA5', 
        col_le: 'LE_ERA5'
    })
    
    df_icos = df_icos[['Site', 'Date', col_et, col_le]].rename(columns={
        col_et: 'ET_ICOS', 
        col_le: 'LE_ICOS'
    })
    
    # On fusionne sur Site et Date pour n'avoir que les points communs
    df_comp = pd.merge(df_era5, df_icos, on=['Site', 'Date'], how='inner')
    
    # On enlève les NaN
    df_comp = df_comp.dropna(subset=['LE_ERA5', 'LE_ICOS'])
    
    if df_comp.empty:
        print("⚠️ Aucune donnée concordante trouvée.")
        return
        
    # --- Calcul des métriques pour LE ---
    modele_era5 = df_comp['LE_ERA5'].values
    modele_icos = df_comp['LE_ICOS'].values
    
    rmse = np.sqrt(mean_squared_error(modele_icos, modele_era5))
    biais = np.mean(modele_era5 - modele_icos)
    r2 = r2_score(modele_icos, modele_era5)
    
    print(f"\n📊 COMPARAISON TTME: Forçage ERA5 vs Forçage ICOS Pur")
    print(f"Nombre de points communs : {len(df_comp)}")
    print(f"RMSE  : {rmse:.2f} W/m²")
    print(f"Biais : {biais:.2f} W/m²")
    print(f"R²    : {r2:.3f}")
    
    # --- Génération du graphique Scatter ---
    plt.figure(figsize=(9, 8))
    
    # Utiliser des couleurs différentes par site
    sites = df_comp['Site'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sites)))
    
    for site, color in zip(sites, colors):
        subset = df_comp[df_comp['Site'] == site]
        plt.scatter(subset['LE_ICOS'], subset['LE_ERA5'], 
                    alpha=0.8, edgecolors='white', s=80, label=site, color=color)
    
    # Ligne 1:1
    min_val = min(np.min(modele_icos), np.min(modele_era5))
    max_val = max(np.max(modele_icos), np.max(modele_era5))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="1:1 (Idéal)")
    
    plt.title("Sensibilité du Modèle TTME aux forçages météorologiques\nERA5 (Grille 9km) vs ICOS (Station in-situ)", fontsize=14, pad=20)
    plt.xlabel("Chaleur Latente (LE) avec Météo ICOS pure (W/m²)", fontsize=12)
    plt.ylabel("Chaleur Latente (LE) avec Météo ERA5 (W/m²)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Ajouter le texte des métriques
    textstr = '\n'.join((
        f'N = {len(df_comp)} dates',
        f'RMSE = {rmse:.1f} W/m²',
        f'Biais = {biais:.1f} W/m²',
        f'R² = {r2:.2f}'))
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
            
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), framealpha=0.9)
    
    output_png = r"C:\Users\a951444\Workspace\projet-satellite\Outputs\Scatter_TTME_ERA5_vs_ICOS.png"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\n✅ Graphique sauvegardé : {output_png}")
    
    # Sauvegarde des métriques dans un CSV
    df_comp.to_csv(r"C:\Users\a951444\Workspace\projet-satellite\Outputs\Donnees_Comparaison_ERA5_ICOS.csv", index=False)

if __name__ == "__main__":
    main()
