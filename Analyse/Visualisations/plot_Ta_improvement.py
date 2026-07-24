import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys

sys.path.append('c:/Users/a951444/Workspace/projet-satellite')

# Set style
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1c1c22", "figure.facecolor": "#1c1c22", "grid.color": "#2d2d35", "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white", "ytick.color": "white"})

# Load data
df_pure = pd.read_csv('c:/Users/a951444/Workspace/projet-satellite/Outputs/Toutes_Comparaisons/Comparaison_ET_Landsat_vs_PureICOS.csv')
df_era5 = pd.read_csv('c:/Users/a951444/Workspace/projet-satellite/Outputs/Resultats_ET_TTME_ERA5.csv')
df_ds = pd.read_csv('c:/Users/a951444/Workspace/projet-satellite/Outputs/Resultats_ET_TTME_ERA5_DS.csv')

def get_col(df, name):
    for c in df.columns:
        if name in c: return c

c_Ta = get_col(df_pure, 'Ta (')

df1 = pd.merge(df_pure[['Site', 'Date', c_Ta]], df_era5[['Site', 'Date', c_Ta]].rename(columns={c_Ta: 'Ta_ERA5'}), on=['Site', 'Date'])
df2 = pd.merge(df1, df_ds[['Site', 'Date', c_Ta]].rename(columns={c_Ta: 'Ta_DS'}), on=['Site', 'Date'])
df2 = df2.dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Validation du modèle de Downscaling (Random Forest) pour la Température de l\'Air', fontsize=18, fontweight='bold', color='#ffffff', y=0.98)

min_val = min(df2[c_Ta].min(), df2['Ta_ERA5'].min(), df2['Ta_DS'].min()) - 2
max_val = max(df2[c_Ta].max(), df2['Ta_ERA5'].max(), df2['Ta_DS'].max()) + 2

# Plot 1: ERA5 Brute
sns.scatterplot(x=c_Ta, y='Ta_ERA5', data=df2, ax=ax1, color='#ef476f', alpha=0.7, s=80, edgecolor='white', linewidth=0.5)
ax1.plot([min_val, max_val], [min_val, max_val], 'w--', linewidth=2, alpha=0.8)
ax1.set_xlim(min_val, max_val)
ax1.set_ylim(min_val, max_val)
ax1.set_title('ERA5 brute (Résolution 10.5 km)', fontsize=14, pad=15)
ax1.set_xlabel('Ta mesurée à la station ICOS (°C)', fontsize=12)
ax1.set_ylabel('Ta ERA5 brute (°C)', fontsize=12)

r2_1 = r2_score(df2[c_Ta], df2['Ta_ERA5'])
rmse_1 = np.sqrt(mean_squared_error(df2[c_Ta], df2['Ta_ERA5']))
bias_1 = np.mean(df2['Ta_ERA5'] - df2[c_Ta])
textstr1 = f'R² = {r2_1:.2f}\nRMSE = {rmse_1:.2f} °C\nBiais = {bias_1:.2f} °C'
props = dict(boxstyle='round,pad=0.5', facecolor='#2b2b36', alpha=0.9, edgecolor='#ef476f')
ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=props, color='white')

# Plot 2: ERA5 DS
sns.scatterplot(x=c_Ta, y='Ta_DS', data=df2, ax=ax2, color='#06d6a0', alpha=0.7, s=80, edgecolor='white', linewidth=0.5)
ax2.plot([min_val, max_val], [min_val, max_val], 'w--', linewidth=2, alpha=0.8)
ax2.set_xlim(min_val, max_val)
ax2.set_ylim(min_val, max_val)
ax2.set_title('ERA5 Downscalé (Modèle RF à 1 km)', fontsize=14, pad=15)
ax2.set_xlabel('Ta mesurée à la station ICOS (°C)', fontsize=12)
ax2.set_ylabel('Ta ERA5 DS (°C)', fontsize=12)

r2_2 = r2_score(df2[c_Ta], df2['Ta_DS'])
rmse_2 = np.sqrt(mean_squared_error(df2[c_Ta], df2['Ta_DS']))
bias_2 = np.mean(df2['Ta_DS'] - df2[c_Ta])
textstr2 = f'R² = {r2_2:.2f}\nRMSE = {rmse_2:.2f} °C\nBiais = {bias_2:.2f} °C'
props2 = dict(boxstyle='round,pad=0.5', facecolor='#2b2b36', alpha=0.9, edgecolor='#06d6a0')
ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=props2, color='white')

plt.tight_layout(rect=[0, 0, 1, 0.93])

output_path = 'c:/Users/a951444/Workspace/projet-satellite/Outputs/Comparaison_ET/Comparaison_Ta_ERA5_vs_DS.png'
os.makedirs('c:/Users/a951444/Workspace/projet-satellite/Outputs/Comparaison_ET', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Graphique sauvegardé: {output_path}")
