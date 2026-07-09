import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error

# Create Outputs dir if missing
os.makedirs("Outputs_performances", exist_ok=True)

# 1. Charger les données
print("Chargement des donnees ET...")
try:
    df_icos = pd.read_csv("Outputs/Resultats_ET_TTME.csv")[['Site', 'Date', 'ET_pixel (mm/h)']]
    df_era5 = pd.read_csv("Outputs/Resultats_ET_TTME_ERA5.csv")[['Site', 'Date', 'ET_pixel (mm/h)']]
    df_era5_b10 = pd.read_csv("Outputs/Resultats_ET_TTME_ERA5_B10.csv")[['Site', 'Date', 'ET_pixel (mm/h)']]
    df_pure_icos = pd.read_csv("Outputs/Toutes_Comparaisons/Comparaison_ET_Landsat_vs_PureICOS.csv")[['Site', 'Date', 'ET_pure_ICOS (mm/h)']]
except Exception as e:
    print(f"Erreur lors du chargement des fichiers : {e}")
    exit(1)

# Renommer pour la fusion
df_icos = df_icos.rename(columns={'ET_pixel (mm/h)': 'ET_ICOS_LST_Landsat'})
df_era5 = df_era5.rename(columns={'ET_pixel (mm/h)': 'ET_ERA5_LST_Landsat'})
df_era5_b10 = df_era5_b10.rename(columns={'ET_pixel (mm/h)': 'ET_ERA5_LST_B10'})
df_pure_icos = df_pure_icos.rename(columns={'ET_pure_ICOS (mm/h)': 'ET_Pure_ICOS'})

# Fusionner les DataFrames
df_merged = pd.merge(df_icos, df_era5, on=['Site', 'Date'], how='inner')
df_merged = pd.merge(df_merged, df_era5_b10, on=['Site', 'Date'], how='inner')
df_merged = pd.merge(df_merged, df_pure_icos, on=['Site', 'Date'], how='inner')

# Nettoyage
df_valid = df_merged.dropna().copy()
print(f"{len(df_valid)} points de comparaison valides trouves.")

if df_valid.empty:
    print("Aucune donnee de comparaison.")
    exit(0)

# Configuration des modèles
MODELES = {
    'ICOS (LST Landsat)': {'col': 'ET_ICOS_LST_Landsat', 'color': '#2ca02c'},
    'ERA5 (LST Landsat)': {'col': 'ET_ERA5_LST_Landsat', 'color': '#ff7f0e'},
    'ERA5 (LST B10)':     {'col': 'ET_ERA5_LST_B10',     'color': '#d62728'}
}

# Calculer les erreurs
for nom, cfg in MODELES.items():
    col_erreur = f'Erreur_{nom.replace(" ", "_")}'
    df_valid[col_erreur] = df_valid[cfg['col']] - df_valid['ET_Pure_ICOS']

# ==========================================
# Graphique 1 : Bar Chart des MAE et RMSE par Site
# ==========================================
print("Génération du graphique des performances par site...")
metrics_site = []
for site in df_valid['Site'].unique():
    subset = df_valid[df_valid['Site'] == site]
    if len(subset) == 0: continue
    
    for nom, cfg in MODELES.items():
        col_erreur = f'Erreur_{nom.replace(" ", "_")}'
        rmse = np.sqrt(mean_squared_error(subset['ET_Pure_ICOS'], subset[cfg['col']]))
        mae = subset[col_erreur].abs().mean()
        metrics_site.append({'Site': site, 'Modele': nom, 'RMSE': rmse, 'MAE': mae})

if metrics_site:
    df_metrics = pd.DataFrame(metrics_site)
    palette_bar = {nom: cfg['color'] for nom, cfg in MODELES.items()}
    
    plt.figure(figsize=(16, 6))
    sns.set_theme(style="whitegrid")
    
    # Subplot 1: MAE par site
    plt.subplot(1, 2, 1)
    sns.barplot(x='Site', y='MAE', hue='Modele', data=df_metrics, palette=palette_bar)
    plt.title('MAE (mm/h) par Site', fontsize=14)
    plt.ylabel('MAE (mm/h)', fontsize=11)
    plt.xticks(rotation=45)
    
    # Subplot 2: RMSE par site
    plt.subplot(1, 2, 2)
    sns.barplot(x='Site', y='RMSE', hue='Modele', data=df_metrics, palette=palette_bar)
    plt.title('RMSE (mm/h) par Site', fontsize=14)
    plt.ylabel('RMSE (mm/h)', fontsize=11)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    output_img_site = "Outputs_performances/Performances_ET_par_Site.png"
    plt.savefig(output_img_site, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé dans : {output_img_site}")

# ==========================================
# Graphique 2 : RMSE, MAE et Biais Global par Modèle
# ==========================================
print("\nGénération du graphique global RMSE, MAE et Biais...")
performance_global_data = []
for nom, cfg in MODELES.items():
    col_erreur = f'Erreur_{nom.replace(" ", "_")}'
    rmse_val = np.sqrt(mean_squared_error(df_valid['ET_Pure_ICOS'], df_valid[cfg['col']]))
    mae_val = df_valid[col_erreur].abs().mean()
    performance_global_data.append({
        'Modele': nom, 
        'RMSE Global': rmse_val,
        'MAE Global': mae_val
    })

if performance_global_data:
    df_perf_global = pd.DataFrame(performance_global_data)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    ordre_modeles = df_perf_global['Modele'].tolist()
    palette_global = {nom: cfg['color'] for nom, cfg in MODELES.items()}
    
    # 1. Barplot RMSE
    sns.barplot(x='Modele', y='RMSE Global', data=df_perf_global, order=ordre_modeles, palette=palette_global, ax=ax1)
    ax1.set_title('RMSE Global (mm/h)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE (mm/h)', fontsize=12)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontweight='bold')
                     
    # 2. Barplot MAE
    sns.barplot(x='Modele', y='MAE Global', data=df_perf_global, order=ordre_modeles, palette=palette_global, ax=ax2)
    ax2.set_title('MAE Globale (mm/h)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE (mm/h)', fontsize=12)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='bottom', fontweight='bold')
                     
    # 3. Boxplot Biais
    erreur_cols = [f'Erreur_{nom.replace(" ", "_")}' for nom in MODELES.keys()]
    df_melted = df_valid.melt(id_vars=['Site'], value_vars=erreur_cols, var_name='Modele', value_name='Erreur (mm/h)')
    df_melted['Modele'] = df_melted['Modele'].str.replace('Erreur_', '').str.replace('_', ' ')
    
    # Ensure correct keys for palette in boxplot by removing Erreur_ prefix and matching MODELES keys
    boxplot_palette = {nom: MODELES[nom]['color'] for nom in MODELES.keys()}
    
    sns.boxplot(x='Modele', y='Erreur (mm/h)', hue='Modele', data=df_melted, order=ordre_modeles, palette=boxplot_palette, legend=False, ax=ax3)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax3.set_title('Distribution des Erreurs (Biais)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_rmse_mae_bias_global = "Outputs_performances/Performances_ET_Globales.png"
    plt.savefig(output_rmse_mae_bias_global, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique global sauvegardé dans : {output_rmse_mae_bias_global}")
