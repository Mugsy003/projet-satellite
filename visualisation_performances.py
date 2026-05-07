import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error

# Create Outputs dir if missing
os.makedirs("Outputs", exist_ok=True)

# 1. Charger les données
csv_path = "Outputs/Validation_Saisonniere_LST.csv"
if not os.path.exists(csv_path):
    csv_path = "Outputs/Validation_Saisonniere_LST_2526.csv"
    if not os.path.exists(csv_path):
        print("Fichier CSV de validation introuvable.")
        exit(1)

print(f"Chargement des données depuis {csv_path}...")
df = pd.read_csv(csv_path)

# Nettoyage des données "N/A"
df.replace("N/A", np.nan, inplace=True)

# S'assurer que les colonnes numériques sont bien des nombres
cols_num = ['ICOS_LST (°C)', 'Temperature_GOL (°C)', 'LST_Sat_DMS (°C)', 'LST_Sat_TsHARP (°C)']
for c in cols_num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Combiner ICOS et GOL en une seule colonne "Vérité Terrain"
# On prend ICOS en priorité, et si c'est NaN on prend GOL
if 'Temperature_GOL (°C)' in df.columns and 'ICOS_LST (°C)' in df.columns:
    df['Ground_LST (°C)'] = df['ICOS_LST (°C)'].fillna(df['Temperature_GOL (°C)'])
elif 'ICOS_LST (°C)' in df.columns:
    df['Ground_LST (°C)'] = df['ICOS_LST (°C)']
elif 'Temperature_GOL (°C)' in df.columns:
    df['Ground_LST (°C)'] = df['Temperature_GOL (°C)']
else:
    print("Aucune colonne de vérité terrain (ICOS ou GOL) trouvée.")
    exit(1)

# Filtrer les lignes où on a bien une vérité terrain (ICOS ou GOL) et une prédiction DMS
df_valid = df.dropna(subset=['Ground_LST (°C)', 'LST_Sat_DMS (°C)']).copy()

if df_valid.empty:
    print("Aucune donnée de comparaison valide trouvée (Vérité Terrain & DMS LST) dans le CSV.")
    exit(0)

print(f"{len(df_valid)} points de comparaison trouvés (ICOS ou GOL).")

# Recalculer la VRAIE erreur (signée, Prédiction - Vérité Terrain)
df_valid['Erreur_DMS'] = df_valid['LST_Sat_DMS (°C)'] - df_valid['Ground_LST (°C)']
df_valid['Erreur_TsHARP'] = df_valid['LST_Sat_TsHARP (°C)'] - df_valid['Ground_LST (°C)']

plt.figure(figsize=(16, 12))
sns.set_theme(style="whitegrid")

# ==========================================
# Graphique 1 : Scatter Plot 1:1
# ==========================================
plt.subplot(2, 2, 1)
plt.scatter(df_valid['Ground_LST (°C)'], df_valid['LST_Sat_DMS (°C)'], alpha=0.6, label='DMS', color='#1f77b4', edgecolor='w', s=60)
plt.scatter(df_valid['Ground_LST (°C)'], df_valid['LST_Sat_TsHARP (°C)'], alpha=0.6, label='TsHARP', color='#d62728', edgecolor='w', s=60)

# Ligne parfaite 1:1
min_val = min(df_valid['Ground_LST (°C)'].min(), df_valid['LST_Sat_DMS (°C)'].min())
max_val = max(df_valid['Ground_LST (°C)'].max(), df_valid['LST_Sat_DMS (°C)'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Parfait (1:1)')

plt.xlabel('Vérité Terrain : ICOS ou GOL (°C)', fontsize=11)
plt.ylabel('Température Prédite par Satellite (°C)', fontsize=11)
plt.title('Nuage de points : Prédictions vs Réalité', fontsize=14)
plt.legend()

# ==========================================
# Graphique 2 : Boxplot des Biais
# ==========================================
plt.subplot(2, 2, 2)
df_melted = df_valid.melt(id_vars=['Site'], value_vars=['Erreur_DMS', 'Erreur_TsHARP'], 
                          var_name='Modèle', value_name='Erreur (°C)')

df_melted['Modèle'] = df_melted['Modèle'].str.replace('Erreur_', '')
df_melted = df_melted.dropna(subset=['Erreur (°C)'])

sns.boxplot(x='Modèle', y='Erreur (°C)', data=df_melted, palette=['#1f77b4', '#d62728'])
plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.title('Distribution des Erreurs (Biais: Prédit - Terrain)', fontsize=14)
plt.ylabel('Biais Directionnel (°C)', fontsize=11)

# ==========================================
# Graphique 3 : Bar Chart des RMSE par Site
# ==========================================
plt.subplot(2, 2, 3)
rmse_data = []
for site in df_valid['Site'].unique():
    subset = df_valid[df_valid['Site'] == site]
    
    # RMSE DMS
    sub_dms = subset.dropna(subset=['LST_Sat_DMS (°C)', 'Ground_LST (°C)'])
    if len(sub_dms) > 0:
        rmse_dms = np.sqrt(mean_squared_error(sub_dms['Ground_LST (°C)'], sub_dms['LST_Sat_DMS (°C)']))
        rmse_data.append({'Site': site, 'Modèle': 'DMS', 'RMSE': rmse_dms})
        
    # RMSE TsHARP
    sub_tsharp = subset.dropna(subset=['LST_Sat_TsHARP (°C)', 'Ground_LST (°C)'])
    if len(sub_tsharp) > 0:
        rmse_tsharp = np.sqrt(mean_squared_error(sub_tsharp['Ground_LST (°C)'], sub_tsharp['LST_Sat_TsHARP (°C)']))
        rmse_data.append({'Site': site, 'Modèle': 'TsHARP', 'RMSE': rmse_tsharp})

if rmse_data:
    df_rmse = pd.DataFrame(rmse_data)
    sns.barplot(x='Site', y='RMSE', hue='Modèle', data=df_rmse, palette=['#1f77b4', '#d62728'])
    plt.title('RMSE par Site Pilote', fontsize=14)
    plt.ylabel('RMSE (°C)', fontsize=11)
    plt.xticks(rotation=45)

# ==========================================
# Graphique 4 : Série temporelle des biais
# ==========================================
plt.subplot(2, 2, 4)
if 'Date_Satellite' in df_valid.columns:
    df_valid['Date_Satellite'] = pd.to_datetime(df_valid['Date_Satellite'])
    df_sorted = df_valid.sort_values('Date_Satellite')

    plt.plot(df_sorted['Date_Satellite'], df_sorted['Erreur_DMS'], marker='o', linestyle='-', alpha=0.7, label='Biais DMS', color='#1f77b4')
    plt.plot(df_sorted['Date_Satellite'], df_sorted['Erreur_TsHARP'], marker='s', linestyle='-', alpha=0.7, label='Biais TsHARP', color='#d62728')
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Évolution temporelle des Biais', fontsize=14)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Biais (°C)', fontsize=11)
    plt.legend()
    plt.xticks(rotation=45)

plt.tight_layout()
output_img = "Outputs/Performances_Modeles.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé avec succès dans : {output_img}")
