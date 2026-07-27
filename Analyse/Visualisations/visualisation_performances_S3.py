import os
import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from config import FILTRER_DATES_VISU, DATE_DEBUT_VISU, DATE_FIN_VISU

# Create Outputs dir if missing
os.makedirs("Outputs_performances", exist_ok=True)

# 1. Charger les données Sentinel-3
csv_path = "Outputs/Resultats_CSV/Validation_Saisonniere_LST_S3.csv"
if not os.path.exists(csv_path):
    print("Fichier CSV de validation Sentinel-3 introuvable.")
    print("Veuillez d'abord exécuter : python comparaison_S3_ICOS.py")
    exit(1)

print(f"Chargement des données S3 depuis {csv_path}...")
df = pd.read_csv(csv_path)

# Configuration du filtrage de temps
FILTRER_DATES = FILTRER_DATES_VISU
DATE_DEBUT = DATE_DEBUT_VISU
DATE_FIN = DATE_FIN_VISU

# Nettoyage des données "N/A"
df.replace("N/A", np.nan, inplace=True)

# S'assurer que les colonnes numériques sont bien des nombres
cols_num = [
    'ICOS_LST (°C)', 'Temp_GOL (°C)',
    'LST_S3_1km (°C)', 'LST_S3_DMS_300m (°C)', 'LST_S3_Fusion_10m (°C)'
]
for c in cols_num:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# Combiner ICOS et GOL en une seule colonne "Vérité Terrain"
if 'Temp_GOL (°C)' in df.columns and 'ICOS_LST (°C)' in df.columns:
    df['Ground_LST (°C)'] = df['ICOS_LST (°C)'].fillna(df['Temp_GOL (°C)'])
elif 'ICOS_LST (°C)' in df.columns:
    df['Ground_LST (°C)'] = df['ICOS_LST (°C)']
elif 'Temp_GOL (°C)' in df.columns:
    df['Ground_LST (°C)'] = df['Temp_GOL (°C)']
else:
    print("Aucune colonne de vérité terrain (ICOS ou GOL) trouvée.")
    exit(1)

# Filtrer : au moins une prédiction ET une vérité terrain
pred_cols = ['LST_S3_1km (°C)', 'LST_S3_DMS_300m (°C)', 'LST_S3_Fusion_10m (°C)']
existing_pred_cols = [c for c in pred_cols if c in df.columns]
df_valid = df.dropna(subset=['Ground_LST (°C)']).copy()
df_valid = df_valid[df_valid[existing_pred_cols].notna().any(axis=1)]

# Filtrage par intervalle de temps
if 'Date_Satellite' in df_valid.columns:
    df_valid['Date_Satellite'] = pd.to_datetime(df_valid['Date_Satellite'])
    if FILTRER_DATES:
        print(f"\n📅 Filtrage de l'intervalle de temps activé : {DATE_DEBUT} au {DATE_FIN}")
        n_avant = len(df_valid)
        if DATE_DEBUT:
            df_valid = df_valid[df_valid['Date_Satellite'] >= pd.to_datetime(DATE_DEBUT)]
        if DATE_FIN:
            df_valid = df_valid[df_valid['Date_Satellite'] <= pd.to_datetime(DATE_FIN)]
        print(f"   - {len(df_valid)} points conservés sur {n_avant} après filtrage des dates.")

if df_valid.empty:
    print("Aucune donnée de comparaison valide trouvée.")
    exit(0)

print(f"{len(df_valid)} points de comparaison trouvés (ICOS ou GOL).")

# Configuration des modèles Sentinel-3
MODELES = {
    'Raw 1km':           {'col': 'LST_S3_1km (°C)',          'color': '#1f77b4', 'marker': 'o'},
    'Sharpened DMS 300m':{'col': 'LST_S3_DMS_300m (°C)',      'color': '#ff7f0e', 'marker': 's'},
    'Fusion S3+S2 10m':  {'col': 'LST_S3_Fusion_10m (°C)',   'color': '#2ca02c', 'marker': '^'},
}

# Filtrer uniquement les modèles dont les colonnes existent
modeles_actifs = {k: v for k, v in MODELES.items() if v['col'] in df_valid.columns}

# Calculer les erreurs
for nom, cfg in modeles_actifs.items():
    col_erreur = f'Erreur_{nom.replace(" ", "_")}'
    df_valid[col_erreur] = df_valid[cfg['col']] - df_valid['Ground_LST (°C)']

# ==========================================
# FILTRAGE DES OUTLIERS (IMPORTANT POUR S3 - VIRER LES NUAGES)
# ==========================================
FILTRER_OUTLIERS = True
OUTLIER_METHODE = "IQR"
OUTLIER_SEUIL = 1.5  # Seuil IQR (très robuste contre les anomalies nuageuses froides)

if FILTRER_OUTLIERS and not df_valid.empty:
    outlier_indices = set()
    print(f"\n🔍 Filtrage des points outliers activé (Méthode: {OUTLIER_METHODE}, Seuil: {OUTLIER_SEUIL})")
    
    for nom, cfg in modeles_actifs.items():
        col_erreur = f'Erreur_{nom.replace(" ", "_")}'
        if col_erreur in df_valid.columns:
            err = df_valid[col_erreur].dropna()
            if len(err) > 0:
                if OUTLIER_METHODE == "IQR":
                    q1 = err.quantile(0.25)
                    q3 = err.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - OUTLIER_SEUIL * iqr
                    upper_bound = q3 + OUTLIER_SEUIL * iqr
                else:
                    continue
                
                cond = (df_valid[col_erreur] < lower_bound) | (df_valid[col_erreur] > upper_bound)
                outliers_mod = df_valid[cond].index
                if len(outliers_mod) > 0:
                    print(f"   - {nom} : {len(outliers_mod)} outliers détectés (Erreur hors de [{lower_bound:.2f}°C, {upper_bound:.2f}°C])")
                    outlier_indices.update(outliers_mod)
                    
    if outlier_indices:
        print(f"❌ Suppression de {len(outlier_indices)} points aberrants (principalement nuages) sur {len(df_valid)}.")
        df_outliers = df_valid.loc[list(outlier_indices)].copy()
        df_outliers.to_csv("Outputs_performances/Outliers_Retires_S3.csv", index=False)
        
        df_valid = df_valid.drop(index=list(outlier_indices))
        print(f"✅ Reste {len(df_valid)} points propres pour l'évaluation et les graphiques.")

# ==========================================
# GÉNÉRATION DES GRAPHQUES ET DU TABLEAU
# ==========================================
plt.figure(figsize=(18, 12))
sns.set_theme(style="whitegrid")

# Graphique 1 : Scatter Plot 1:1
plt.subplot(2, 2, 1)
all_vals = []
for nom, cfg in modeles_actifs.items():
    mask = df_valid[cfg['col']].notna()
    if mask.any():
        plt.scatter(df_valid.loc[mask, 'Ground_LST (°C)'], df_valid.loc[mask, cfg['col']], 
                    alpha=0.7, label=nom, color=cfg['color'], marker=cfg['marker'], edgecolor='w', s=60)
        all_vals.extend(df_valid.loc[mask, cfg['col']].tolist())
        all_vals.extend(df_valid.loc[mask, 'Ground_LST (°C)'].tolist())

if all_vals:
    min_val = np.nanmin(all_vals)
    max_val = np.nanmax(all_vals)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Parfait (1:1)')

plt.xlabel('Vérité Terrain : ICOS ou GOL (°C)', fontsize=11)
plt.ylabel('Température Prédite par Sentinel-3 (°C)', fontsize=11)
plt.title('Predictions vs Réalité (S3)', fontsize=14, fontweight='bold')
plt.legend(fontsize=9)

# Graphique 2 : Boxplot des Biais
plt.subplot(2, 2, 2)
erreur_cols = [f'Erreur_{nom.replace(" ", "_")}' for nom in modeles_actifs.keys() if f'Erreur_{nom.replace(" ", "_")}' in df_valid.columns]
df_melted = df_valid.melt(id_vars=['Site'], value_vars=erreur_cols, var_name='Modele', value_name='Erreur (°C)')
df_melted['Modele'] = df_melted['Modele'].str.replace('Erreur_', '').str.replace('_', ' ')
df_melted = df_melted.dropna(subset=['Erreur (°C)'])

palette_box = [cfg['color'] for nom, cfg in modeles_actifs.items()]
if not df_melted.empty:
    sns.boxplot(x='Modele', y='Erreur (°C)', hue='Modele', data=df_melted, palette=palette_box, legend=False)
plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.title('Distribution des Biais (Prédit - Terrain) S3', fontsize=14, fontweight='bold')
plt.ylabel('Biais Directionnel (°C)', fontsize=11)
plt.xticks(rotation=15)

# Graphique 3 : Bar Chart des RMSE par Site
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
    plt.title('RMSE Sentinel-3 par Site Pilote', fontsize=14, fontweight='bold')
    plt.ylabel('RMSE (°C)', fontsize=11)
    plt.xticks(rotation=45)

# Graphique 4 : Série temporelle des biais
plt.subplot(2, 2, 4)
if 'Date_Satellite' in df_valid.columns:
    df_sorted = df_valid.sort_values('Date_Satellite')
    for nom, cfg in modeles_actifs.items():
        col_erreur = f'Erreur_{nom.replace(" ", "_")}'
        if col_erreur in df_sorted.columns:
            mask = df_sorted[col_erreur].notna()
            if mask.any():
                plt.plot(df_sorted.loc[mask, 'Date_Satellite'], df_sorted.loc[mask, col_erreur], 
                         marker=cfg['marker'], linestyle='-', alpha=0.7, label=f'Biais {nom}', color=cfg['color'], markersize=5)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
    plt.title('Évolution temporelle des Biais (S3)', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Biais (°C)', fontsize=11)
    plt.legend(fontsize=9)
    plt.xticks(rotation=45)

plt.tight_layout()
output_img_global = "Outputs_performances/Performances_Modeles_S3_Global.png"
plt.savefig(output_img_global, dpi=300, bbox_inches='tight')
plt.close()
print(f"Graphique global sauvegardé dans : {output_img_global}")

# ==========================================
# Tableau récapitulatif RMSE + Biais Global
# ==========================================
print("\n" + "="*65)
print("📊 RAPPORTS DE PERFORMANCES S3 GLOBALES (APRES IQR OUTLIERS)")
print("="*65)
for nom, cfg in modeles_actifs.items():
    col_erreur = f'Erreur_{nom.replace(" ", "_")}'
    if col_erreur in df_valid.columns:
        erreurs = df_valid[col_erreur].dropna()
        if len(erreurs) > 0:
            rmse_global = np.sqrt(np.mean(erreurs**2))
            biais_moyen = erreurs.mean()
            mae = erreurs.abs().mean()
            print(f"  {nom:20s} | N={len(erreurs):3d} | RMSE={rmse_global:.2f}°C | Biais={biais_moyen:+.2f}°C | MAE={mae:.2f}°C")

# ==========================================
# Génération des graphiques RMSE, MAE, Bias Globaux (Image Séparée)
# ==========================================
performance_global_data = []
for nom, cfg in modeles_actifs.items():
    sub = df_valid.dropna(subset=[cfg['col'], 'Ground_LST (°C)'])
    if len(sub) > 0:
        col_erreur = f'Erreur_{nom.replace(" ", "_")}'
        rmse_val = np.sqrt(mean_squared_error(sub['Ground_LST (°C)'], sub[cfg['col']]))
        mae_val = sub[col_erreur].abs().mean()
        performance_global_data.append({
            'Modele': nom, 
            'RMSE Global (°C)': rmse_val,
            'MAE Global (°C)': mae_val
        })

if performance_global_data:
    df_perf_global = pd.DataFrame(performance_global_data).sort_values(by='RMSE Global (°C)')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    ordre_modeles = df_perf_global['Modele'].tolist()
    palette_global = {row['Modele']: modeles_actifs[row['Modele']]['color'] for _, row in df_perf_global.iterrows()}
    palette_list_aligned = [modeles_actifs[m]['color'] for m in ordre_modeles]
    
    # 1. RMSE
    sns.barplot(x='Modele', y='RMSE Global (°C)', data=df_perf_global, order=ordre_modeles, palette=palette_global, ax=ax1)
    ax1.set_title('RMSE Global (S3)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RMSE (°C)', fontsize=12)
    ax1.tick_params(axis='x', rotation=15)
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.2f}°C", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
                    
    # 2. MAE
    sns.barplot(x='Modele', y='MAE Global (°C)', data=df_perf_global, order=ordre_modeles, palette=palette_global, ax=ax2)
    ax2.set_title('MAE Globale (S3)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE (°C)', fontsize=12)
    ax2.tick_params(axis='x', rotation=15)
    for p in ax2.patches:
        ax2.annotate(f"{p.get_height():.2f}°C", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), textcoords='offset points')
                    
    # 3. Boxplot Biais
    if not df_melted.empty:
        sns.boxplot(x='Modele', y='Erreur (°C)', hue='Modele', data=df_melted, order=ordre_modeles, palette=palette_list_aligned, legend=False, ax=ax3)
    ax3.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax3.set_title('Biais (S3)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Biais (°C)', fontsize=12)
    ax3.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    output_rmse_mae_bias_global = "Outputs_performances/Performances_Modeles_S3_RMSE_MAE_Bias_Global.png"
    plt.savefig(output_rmse_mae_bias_global, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graphique de performances globales S3 sauvegardé dans : {output_rmse_mae_bias_global}\n")

# ==========================================
# Génération des graphiques par Site
# ==========================================
for site in df_valid['Site'].unique():
    df_site = df_valid[df_valid['Site'] == site].copy()
    if len(df_site) == 0:
        continue
    try:
        plt.figure(figsize=(18, 5))
        
        # 1. Scatter Plot
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
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Parfait (1:1)')
        plt.xlabel('Vérité Terrain (°C)', fontsize=11)
        plt.ylabel('Prédiction Sentinel-3 (°C)', fontsize=11)
        plt.title(f'Predictions vs Réalité - {site}', fontsize=13, fontweight='bold')
        plt.legend(fontsize=8)
        
        # 2. Boxplot biais
        plt.subplot(1, 3, 2)
        df_melted_site = df_site.melt(id_vars=['Site'], value_vars=erreur_cols, var_name='Modele', value_name='Erreur (°C)')
        df_melted_site['Modele'] = df_melted_site['Modele'].str.replace('Erreur_', '').str.replace('_', ' ')
        df_melted_site = df_melted_site.dropna(subset=['Erreur (°C)'])
        if not df_melted_site.empty:
            modeles_present = df_melted_site['Modele'].unique()
            palette_site = {nom: modeles_actifs[nom]['color'] for nom in modeles_present if nom in modeles_actifs}
            sns.boxplot(x='Modele', y='Erreur (°C)', hue='Modele', data=df_melted_site, palette=palette_site, legend=False)
        plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
        plt.title(f'Biais - {site}', fontsize=13, fontweight='bold')
        plt.ylabel('Biais (°C)', fontsize=11)
        plt.xticks(rotation=15)
        
        # 3. Série temporelle
        plt.subplot(1, 3, 3)
        if 'Date_Satellite' in df_site.columns:
            df_sorted_site = df_site.sort_values('Date_Satellite')
            for nom, cfg in modeles_actifs.items():
                col_erreur = f'Erreur_{nom.replace(" ", "_")}'
                if col_erreur in df_sorted_site.columns:
                    mask = df_sorted_site[col_erreur].notna()
                    if mask.any():
                        plt.plot(df_sorted_site.loc[mask, 'Date_Satellite'], df_sorted_site.loc[mask, col_erreur], 
                                 marker=cfg['marker'], linestyle='-', alpha=0.8, label=f'Biais {nom}', color=cfg['color'], markersize=5)
            plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
            plt.title(f'Évolution Temporelle - {site}', fontsize=13, fontweight='bold')
            plt.xlabel('Date', fontsize=11)
            plt.ylabel('Biais (°C)', fontsize=11)
            plt.legend(fontsize=8)
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        output_site = f"Outputs_performances/Performances_Modeles_S3_{site}.png"
        plt.savefig(output_site, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graphique spécifique au site sauvegardé dans : {output_site}")
    except Exception as e:
        plt.close()
        print(f"Erreur graphique pour {site} : {e}")
