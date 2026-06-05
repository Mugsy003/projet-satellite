import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- CONFIGURATION ---
PATH_CSV = r"C:\Users\a951444\Workspace\projet-satellite\Outputs\Validation_Saisonniere_LST.csv"
OUTPUT_DIR = r"C:\Users\a951444\Workspace\projet-satellite\Outputs\Analyse_Graphiques"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Chargement et préparation des données
df = pd.read_csv(PATH_CSV)
df['Date_Satellite'] = pd.to_datetime(df['Date_Satellite'])
df = df.sort_values(['Site', 'Date_Satellite'])

# Nettoyage : conversion des colonnes en numérique (cas où il y aurait des "N/A")
cols_num = ['LST_Sat (°C)', 'ICOS_Sol (°C)', 'ICOS_Air (°C)', 'Biais (°C)', 'NDVI']
for col in cols_num:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['LST_Sat (°C)', 'ICOS_Sol (°C)'])

sites = df['Site'].unique()

print(f"📊 Début de l'analyse pour {len(sites)} sites...")

# ==========================================================
# 2. GRAPHIQUES PAR SITE (Séries Temporelles)
# ==========================================================
for site in sites:
    data_site = df[df['Site'] == site]
    
    plt.figure(figsize=(12, 6))
    plt.plot(data_site['Date_Satellite'], data_site['LST_Sat (°C)'], 'o-', label='Satellite (LST Sharpened)', color='red', markersize=8)
    plt.plot(data_site['Date_Satellite'], data_site['ICOS_Sol (°C)'], 's--', label='ICOS Sol (1cm)', color='brown')
    plt.plot(data_site['Date_Satellite'], data_site['ICOS_Air (°C)'], 'x:', label='ICOS Air', color='green', alpha=0.6)
    
    plt.title(f"Évolution des Températures - Site : {site}")
    plt.ylabel("Température (°C)")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}.png"))
    plt.close()

# ==========================================================
# 3. ANALYSE DE CORRÉLATION GLOBALE (Satellite vs Sol)
# ==========================================================
plt.figure(figsize=(8, 8))
sns.scatterplot(data=df, x='ICOS_Sol (°C)', y='LST_Sat (°C)', hue='Site', style='Site', s=100)

# Ligne 1:1 (Identité)
max_val = max(df['LST_Sat (°C)'].max(), df['ICOS_Sol (°C)'].max())
min_val = min(df['LST_Sat (°C)'].min(), df['ICOS_Sol (°C)'].min())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ligne 1:1')

plt.title("Corrélation Globale : Satellite vs Vérité Terrain (Sol)")
plt.xlabel("ICOS Sol 1cm (°C)")
plt.ylabel("DMS Sharpened LST (°C)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Correlation_Globale_Sat_vs_Sol.png"))
plt.close()

# ==========================================================
# 4. ANALYSE DE L'EFFET DU NDVI SUR LE BIAIS
# ==========================================
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='NDVI', y='Biais (°C)', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})

plt.title("Impact de la Végétation (NDVI) sur le Biais (Sat - Sol)")
plt.xlabel("NDVI (Végétation)")
plt.ylabel("Biais Absolu (°C)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Analyse_NDVI_vs_Biais.png"))
plt.close()

# ==========================================================
# 5. CALCUL DES MÉTRIQUES STATISTIQUES
# ==========================================================
stats = []
for site in sites:
    ds = df[df['Site'] == site]
    if len(ds) > 1:
        rmse = np.sqrt(mean_squared_error(ds['ICOS_Sol (°C)'], ds['LST_Sat (°C)']))
        r2 = r2_score(ds['ICOS_Sol (°C)'], ds['LST_Sat (°C)'])
        corr = ds['LST_Sat (°C)'].corr(ds['ICOS_Sol (°C)'])
        
        stats.append({
            "Site": site,
            "N_Images": len(ds),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 2),
            "Correlation": round(corr, 2),
            "Biais_Moyen": round(ds['Biais (°C)'].mean(), 2)
        })

df_stats = pd.DataFrame(stats)
print("\n" + "="*50)
print("📈 RÉSULTATS STATISTIQUES PAR SITE")
print("="*50)
print(df_stats.to_markdown(index=False))

df_stats.to_csv(os.path.join(OUTPUT_DIR, "Statistiques_Validation.csv"), index=False)