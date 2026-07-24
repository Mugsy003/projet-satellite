import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import matplotlib.dates as mdates

# Charger les données de comparaison ERA5
df_era5 = pd.read_csv(r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Toutes_Comparaisons\Comparaison_ET_ICOS_vs_ERA5.csv")

# Charger les données contenant la PURE vérité terrain
df_pure = pd.read_csv(r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Toutes_Comparaisons\Comparaison_ET_Landsat_vs_PureICOS.csv")

# Renommer pour éviter la confusion
df_pure = df_pure[['Site', 'Date', 'ET_pure_ICOS (mm/h)']]

# Merging sur le Site et la Date
df = pd.merge(df_era5, df_pure, on=['Site', 'Date'], how='inner')

# Garder uniquement les lignes où l'on a la vérité terrain
df = df.dropna(subset=['ET_pure_ICOS (mm/h)', 'ET_pixel (mm/h)_ERA5', 'ET_pixel (mm/h)_ERA5_DS'])

# Calculer les VRAIES erreurs absolues par rapport à ICOS PURE (Vérité terrain)
df['Err_ERA5_Brut'] = np.abs(df['ET_pixel (mm/h)_ERA5'] - df['ET_pure_ICOS (mm/h)'])
df['Err_ERA5_DS'] = np.abs(df['ET_pixel (mm/h)_ERA5_DS'] - df['ET_pure_ICOS (mm/h)'])

# Extraire les valeurs NDVI
results = []
for idx, row in df.iterrows():
    site = row['Site']
    date_str = row['Date']
    
    base_dir = f"c:\\Users\\a951444\\Workspace\\projet-satellite\\Outputs\\Serie_Temporelle_{site}\\3_Indices\\TIF_Data"
    ndvi_files = glob.glob(f"{base_dir}\\{date_str}*NDVI.tif")
    
    ndvi_val = np.nan
    if ndvi_files:
        try:
            with rasterio.open(ndvi_files[0]) as src:
                arr = src.read(1)
                h, w = arr.shape
                ndvi_val = arr[int(h/2), int(w/2)]
        except Exception as e:
            pass
            
    results.append({
        'Site': site,
        'Date': pd.to_datetime(date_str),
        'Err_ERA5_Brut': row['Err_ERA5_Brut'],
        'Err_ERA5_DS': row['Err_ERA5_DS'],
        'NDVI': ndvi_val
    })

df_plot = pd.DataFrame(results).dropna()

# Générer un graphique par site
sites = df_plot['Site'].unique()

out_dir = r"c:\Users\a951444\Workspace\projet-satellite\Outputs\Toutes_Comparaisons\NDVI_vs_Erreur"
os.makedirs(out_dir, exist_ok=True)

for site in sites:
    sub = df_plot[df_plot['Site'] == site].sort_values('Date')
    if len(sub) == 0: continue
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color_brut = 'tab:red'
    color_ds = 'tab:blue'
    color_ndvi = 'tab:green'
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Erreur Absolue (vs Pure ICOS) (mm/h)', fontsize=12)
    line1, = ax1.plot(sub['Date'], sub['Err_ERA5_Brut'], marker='o', color=color_brut, label='Erreur ERA5 Brut (vs Pure ICOS)')
    line2, = ax1.plot(sub['Date'], sub['Err_ERA5_DS'], marker='s', color=color_ds, label='Erreur ERA5 DS (vs Pure ICOS)')
    ax1.tick_params(axis='y')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Format x-axis for dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    
    ax2 = ax1.twinx()  
    ax2.set_ylabel('NDVI', color=color_ndvi, fontsize=12)  
    line3, = ax2.plot(sub['Date'], sub['NDVI'], marker='^', color=color_ndvi, linestyle='--', label='NDVI')
    ax2.tick_params(axis='y', labelcolor=color_ndvi)
    ax2.set_ylim(0, 1)
    
    # Legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title(f"Série Temporelle : Erreur Absolue (vs Mesures Sol) vs NDVI - {site}", fontsize=14)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    out_file = os.path.join(out_dir, f"NDVI_vs_Erreur_{site}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    
print("Génération des graphiques (corrigés avec PURE ICOS) terminée.")
