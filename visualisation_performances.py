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

print(f"Chargement des donnees depuis {csv_path}...")
df = pd.read_csv(csv_path)

# Nettoyage des données "N/A"
df.replace("N/A", np.nan, inplace=True)

# S'assurer que les colonnes numériques sont bien des nombres
cols_num = [
    'ICOS_LST (°C)', 'Temperature_GOL (°C)',
    'LST_Sat_DMS (°C)', 'LST_Sat_TsHARP (°C)',
    'LST_Sat_DMS_Fusion (°C)', 'LST_Sat_TsHARP_Fusion (°C)',
    'LST_Sat_DL (°C)'
]
for c in cols_num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Combiner ICOS et GOL en une seule colonne "Vérité Terrain"
if 'Temperature_GOL (°C)' in df.columns and 'ICOS_LST (°C)' in df.columns:
    df['Ground_LST (°C)'] = df['ICOS_LST (°C)'].fillna(df['Temperature_GOL (°C)'])
elif 'ICOS_LST (°C)' in df.columns:
    df['Ground_LST (°C)'] = df['ICOS_LST (°C)']
elif 'Temperature_GOL (°C)' in df.columns:
    df['Ground_LST (°C)'] = df['Temperature_GOL (°C)']
else:
    print("Aucune colonne de verite terrain (ICOS ou GOL) trouvee.")
    exit(1)

# Filtrer : au moins une prédiction ET une vérité terrain
pred_cols = ['LST_Sat_DMS (°C)', 'LST_Sat_TsHARP (°C)', 'LST_Sat_DMS_Fusion (°C)', 'LST_Sat_TsHARP_Fusion (°C)', 'LST_Sat_DL (°C)']
existing_pred_cols = [c for c in pred_cols if c in df.columns]
df_valid = df.dropna(subset=['Ground_LST (°C)']).copy()
df_valid = df_valid[df_valid[existing_pred_cols].notna().any(axis=1)]

if df_valid.empty:
    print("Aucune donnee de comparaison valide trouvee.")
    exit(0)

print(f"{len(df_valid)} points de comparaison trouves (ICOS ou GOL).")

# Configuration des 5 modèles avec couleurs
MODELES = {
    'DMS':            {'col': 'LST_Sat_DMS (°C)',            'color': '#1f77b4', 'marker': 'o'},
    'TsHARP':         {'col': 'LST_Sat_TsHARP (°C)',         'color': '#d62728', 'marker': 's'},
    'DMS Fusion':     {'col': 'LST_Sat_DMS_Fusion (°C)',     'color': '#2ca02c', 'marker': '^'},
    'TsHARP Fusion':  {'col': 'LST_Sat_TsHARP_Fusion (°C)', 'color': '#ff7f0e', 'marker': 'D'},
    'Deep Learning':  {'col': 'LST_Sat_DL (°C)',             'color': '#9467bd', 'marker': 'P'},
}

# Filtrer uniquement les modèles dont les colonnes existent
modeles_actifs = {k: v for k, v in MODELES.items() if v['col'] in df_valid.columns}

# Calculer les erreurs
for nom, cfg in modeles_actifs.items():
    col_erreur = f'Erreur_{nom.replace(" ", "_")}'
    df_valid[col_erreur] = df_valid[cfg['col']] - df_valid['Ground_LST (°C)']

plt.figure(figsize=(18, 12))
sns.set_theme(style="whitegrid")

# ==========================================
# Graphique 1 : Scatter Plot 1:1
# ==========================================
plt.subplot(2, 2, 1)
all_vals = []
for nom, cfg in modeles_actifs.items():
    mask = df_valid[cfg['col']].notna()
    if mask.any():
        plt.scatter(df_valid.loc[mask, 'Ground_LST (°C)'], df_valid.loc[mask, cfg['col']], 
                    alpha=0.6, label=nom, color=cfg['color'], marker=cfg['marker'], edgecolor='w', s=60)
        all_vals.extend(df_valid.loc[mask, cfg['col']].tolist())
        all_vals.extend(df_valid.loc[mask, 'Ground_LST (°C)'].tolist())

if all_vals:
    min_val = np.nanmin(all_vals)
    max_val = np.nanmax(all_vals)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Parfait (1:1)')

plt.xlabel('Verite Terrain : ICOS ou GOL (degC)', fontsize=11)
plt.ylabel('Temperature Predite par Satellite (degC)', fontsize=11)
plt.title('Nuage de points : Predictions vs Realite', fontsize=14)
plt.legend(fontsize=9)

# ==========================================
# Graphique 2 : Boxplot des Biais
# ==========================================
plt.subplot(2, 2, 2)
erreur_cols = [f'Erreur_{nom.replace(" ", "_")}' for nom in modeles_actifs.keys() if f'Erreur_{nom.replace(" ", "_")}' in df_valid.columns]
df_melted = df_valid.melt(id_vars=['Site'], value_vars=erreur_cols, 
                          var_name='Modele', value_name='Erreur (degC)')
df_melted['Modele'] = df_melted['Modele'].str.replace('Erreur_', '').str.replace('_', ' ')
df_melted = df_melted.dropna(subset=['Erreur (degC)'])

palette_box = [cfg['color'] for nom, cfg in modeles_actifs.items()]
if not df_melted.empty:
    sns.boxplot(x='Modele', y='Erreur (degC)', hue='Modele', data=df_melted, palette=palette_box, legend=False)
plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.title('Distribution des Erreurs (Biais: Predit - Terrain)', fontsize=14)
plt.ylabel('Biais Directionnel (degC)', fontsize=11)
plt.xticks(rotation=20)

# ==========================================
# Graphique 3 : Bar Chart des RMSE par Site
# ==========================================
plt.subplot(2, 2, 3)
rmse_data = []
for site in df_valid['Site'].unique():
    subset = df_valid[df_valid['Site'] == site]
    for nom, cfg in modeles_actifs.items():
        sub = subset.dropna(subset=[cfg['col'], 'Ground_LST (°C)'])
        if len(sub) > 0:
            rmse = np.sqrt(mean_squared_error(sub['Ground_LST (°C)'], sub[cfg['col']]))
            rmse_data.append({'Site': site, 'Modele': nom, 'RMSE': rmse})

if rmse_data:
    df_rmse = pd.DataFrame(rmse_data)
    palette_bar = {nom: cfg['color'] for nom, cfg in modeles_actifs.items()}
    sns.barplot(x='Site', y='RMSE', hue='Modele', data=df_rmse, palette=palette_bar)
    plt.title('RMSE par Site Pilote', fontsize=14)
    plt.ylabel('RMSE (degC)', fontsize=11)
    plt.xticks(rotation=45)

# ==========================================
# Graphique 4 : Série temporelle des biais
# ==========================================
plt.subplot(2, 2, 4)
if 'Date_Satellite' in df_valid.columns:
    df_valid['Date_Satellite'] = pd.to_datetime(df_valid['Date_Satellite'])
    df_sorted = df_valid.sort_values('Date_Satellite')

    for nom, cfg in modeles_actifs.items():
        col_erreur = f'Erreur_{nom.replace(" ", "_")}'
        if col_erreur in df_sorted.columns:
            mask = df_sorted[col_erreur].notna()
            if mask.any():
                plt.plot(df_sorted.loc[mask, 'Date_Satellite'], df_sorted.loc[mask, col_erreur], 
                         marker=cfg['marker'], linestyle='-', alpha=0.7, label=f'Biais {nom}', color=cfg['color'], markersize=5)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Evolution temporelle des Biais', fontsize=14)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Biais (degC)', fontsize=11)
    plt.legend(fontsize=9)
    plt.xticks(rotation=45)

plt.tight_layout()
output_img_global = "Outputs/Performances_Modeles_Global.png"
plt.savefig(output_img_global, dpi=300, bbox_inches='tight')
plt.close()
print(f"Graphique global sauvegarde dans : {output_img_global}")

# ==========================================
# Tableau récapitulatif RMSE + Biais Global
# ==========================================
print("\n" + "="*60)
print("TABLEAU RECAPITULATIF DES PERFORMANCES")
print("="*60)
for nom, cfg in modeles_actifs.items():
    col_erreur = f'Erreur_{nom.replace(" ", "_")}'
    if col_erreur in df_valid.columns:
        erreurs = df_valid[col_erreur].dropna()
        if len(erreurs) > 0:
            rmse_global = np.sqrt(np.mean(erreurs**2))
            biais_moyen = erreurs.mean()
            mae = erreurs.abs().mean()
            print(f"  {nom:20s} | N={len(erreurs):3d} | RMSE={rmse_global:.2f}degC | Biais={biais_moyen:+.2f}degC | MAE={mae:.2f}degC")

# ==========================================
# Génération des graphiques par Site
# ==========================================
sites = df_valid['Site'].unique()

for site in sites:
    df_site = df_valid[df_valid['Site'] == site].copy()
    
    if len(df_site) == 0:
        continue
    
    try:
        plt.figure(figsize=(18, 5))
        sns.set_theme(style="whitegrid")
        
        # 1. Scatter Plot par site
        plt.subplot(1, 3, 1)
        for nom, cfg in modeles_actifs.items():
            mask = df_site[cfg['col']].notna()
            if mask.any():
                plt.scatter(df_site.loc[mask, 'Ground_LST (°C)'], df_site.loc[mask, cfg['col']], 
                            alpha=0.8, label=nom, color=cfg['color'], marker=cfg['marker'], edgecolor='w', s=60)

        all_site_vals = []
        for cfg in modeles_actifs.values():
            all_site_vals.extend(df_site[cfg['col']].dropna().tolist())
        all_site_vals.extend(df_site['Ground_LST (°C)'].dropna().tolist())
        
        if all_site_vals:
            min_val = np.nanmin(all_site_vals)
            max_val = np.nanmax(all_site_vals)
            if np.isfinite(min_val) and np.isfinite(max_val):
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Parfait (1:1)')
            
        plt.xlabel('Verite Terrain (degC)', fontsize=11)
        plt.ylabel('Prediction Satellite (degC)', fontsize=11)
        plt.title(f'Predictions vs Realite - {site}', fontsize=14)
        plt.legend(fontsize=8)
        
        # 2. Boxplot des biais par site
        plt.subplot(1, 3, 2)
        erreur_cols_site = [f'Erreur_{nom.replace(" ", "_")}' for nom in modeles_actifs.keys() if f'Erreur_{nom.replace(" ", "_")}' in df_site.columns]
        df_melted_site = df_site.melt(id_vars=['Site'], value_vars=erreur_cols_site, 
                                      var_name='Modele', value_name='Erreur (degC)')
        df_melted_site['Modele'] = df_melted_site['Modele'].str.replace('Erreur_', '').str.replace('_', ' ')
        df_melted_site = df_melted_site.dropna(subset=['Erreur (degC)'])
        
        if not df_melted_site.empty:
            modeles_present = df_melted_site['Modele'].unique()
            palette_site = {nom.replace('_', ' '): cfg['color'] for nom, cfg in modeles_actifs.items() if nom.replace('_', ' ') in modeles_present or nom in modeles_present}
            sns.boxplot(x='Modele', y='Erreur (degC)', hue='Modele', data=df_melted_site, palette=palette_site, legend=False)
        plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
        plt.title(f'Biais (Predit - Terrain) - {site}', fontsize=14)
        plt.ylabel('Biais (degC)', fontsize=11)
        plt.xticks(rotation=20)
        
        # 3. Série temporelle par site
        plt.subplot(1, 3, 3)
        if 'Date_Satellite' in df_site.columns:
            df_site['Date_Satellite'] = pd.to_datetime(df_site['Date_Satellite'])
            df_sorted_site = df_site.sort_values('Date_Satellite')

            for nom, cfg in modeles_actifs.items():
                col_erreur = f'Erreur_{nom.replace(" ", "_")}'
                if col_erreur in df_sorted_site.columns:
                    mask = df_sorted_site[col_erreur].notna()
                    if mask.any():
                        plt.plot(df_sorted_site.loc[mask, 'Date_Satellite'], df_sorted_site.loc[mask, col_erreur], 
                                 marker=cfg['marker'], linestyle='-', alpha=0.8, label=f'Biais {nom}', color=cfg['color'], markersize=5)
                
            plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
            plt.title(f'Evolution Temporelle - {site}', fontsize=14)
            plt.xlabel('Date', fontsize=11)
            plt.ylabel('Biais (degC)', fontsize=11)
            plt.legend(fontsize=8)
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        output_site = f"Outputs/Performances_Modeles_{site}.png"
        plt.savefig(output_site, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphique specifique au site sauvegarde dans : {output_site}")
        
    except Exception as e:
        plt.close()
        print(f"Erreur lors de la generation du graphique pour {site} : {e}")
