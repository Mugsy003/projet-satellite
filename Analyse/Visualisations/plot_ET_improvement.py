import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os

plt.style.use('default')
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa", "figure.facecolor": "white", "grid.color": "#e0e0e0", "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})

df_icos = pd.read_csv('Outputs/Resultats_CSV/Resultats_ET_TTME_ICOS.csv')[['Site', 'Date', 'ET_pixel (mm/h)']].rename(columns={'ET_pixel (mm/h)': 'ET_ICOS'})
df_era5 = pd.read_csv('Outputs/Resultats_CSV/Resultats_ET_TTME_ERA5.csv')[['Site', 'Date', 'ET_pixel (mm/h)']].rename(columns={'ET_pixel (mm/h)': 'ET_ERA5'})
df_ds = pd.read_csv('Outputs/Resultats_CSV/Resultats_ET_TTME_ERA5_DS.csv')[['Site', 'Date', 'ET_pixel (mm/h)']].rename(columns={'ET_pixel (mm/h)': 'ET_DS'})

df = pd.merge(df_icos, df_era5, on=['Site', 'Date'], how='inner')
df = pd.merge(df, df_ds, on=['Site', 'Date'], how='inner')
df = df.dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Amélioration Globale de l\'Évapotranspiration (ET) grâce au Downscaling\n(Évalué contre la Météo ICOS Parfaite)', fontsize=16, fontweight='bold', color='#ffffff', y=0.98)

min_val = 0
max_val = max(df['ET_ICOS'].max(), df['ET_ERA5'].max(), df['ET_DS'].max()) + 0.1

# Plot 1
sns.scatterplot(x='ET_ICOS', y='ET_ERA5', data=df, ax=ax1, color='#ef476f', alpha=0.7, s=80, edgecolor='white', linewidth=0.5)
ax1.plot([min_val, max_val], [min_val, max_val], 'w--', linewidth=2, alpha=0.8)
ax1.set_xlim(min_val, max_val)
ax1.set_ylim(min_val, max_val)
ax1.set_title('ET calculée avec ERA5 brute', fontsize=14, pad=15)
ax1.set_xlabel('ET Météo ICOS Parfaite (mm/h)', fontsize=12)
ax1.set_ylabel('ET avec ERA5 brute (mm/h)', fontsize=12)

r2_1 = r2_score(df['ET_ICOS'], df['ET_ERA5'])
rmse_1 = np.sqrt(mean_squared_error(df['ET_ICOS'], df['ET_ERA5']))
textstr1 = f'R² = {r2_1:.2f}\nRMSE = {rmse_1:.3f} mm/h'
props = dict(boxstyle='round,pad=0.5', facecolor='#ffffff', alpha=0.9, edgecolor='#ef476f')
ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=12, verticalalignment='top', bbox=props, color='black')

# Plot 2
sns.scatterplot(x='ET_ICOS', y='ET_DS', data=df, ax=ax2, color='#06d6a0', alpha=0.7, s=80, edgecolor='white', linewidth=0.5)
ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)
ax2.set_xlim(min_val, max_val)
ax2.set_ylim(min_val, max_val)
ax2.set_title('ET calculée avec ERA5 Downscalé (RF)', fontsize=14, pad=15)
ax2.set_xlabel('ET Météo ICOS Parfaite (mm/h)', fontsize=12)
ax2.set_ylabel('ET avec ERA5 DS (mm/h)', fontsize=12)

r2_2 = r2_score(df['ET_ICOS'], df['ET_DS'])
rmse_2 = np.sqrt(mean_squared_error(df['ET_ICOS'], df['ET_DS']))
textstr2 = f'R² = {r2_2:.2f}\nRMSE = {rmse_2:.3f} mm/h'
props2 = dict(boxstyle='round,pad=0.5', facecolor='#ffffff', alpha=0.9, edgecolor='#06d6a0')
ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=12, verticalalignment='top', bbox=props2, color='black')

plt.tight_layout(rect=[0, 0, 1, 0.93])

output_path = 'Outputs_performances/Comparaison_Globale_ET_Amelioration.png'
os.makedirs('Outputs_performances', exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
