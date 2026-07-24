import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df_icos = pd.read_csv('Outputs/Resultats_ET_TTME_ICOS.csv')
df_pure = pd.read_csv('Outputs/Toutes_Comparaisons/Comparaison_ET_Landsat_vs_PureICOS.csv')

# Merge on Site and Date
df3 = pd.merge(df_icos, df_pure, on=['Site', 'Date']).dropna(subset=['ET_pixel (mm/h)_x', 'ET_pure_ICOS (mm/h)'])

print('Global R2 ICOS vs Pure ICOS:', r2_score(df3['ET_pure_ICOS (mm/h)'], df3['ET_pixel (mm/h)_x']))
print('\n--- Par Site ---')
for s, grp in df3.groupby('Site'):
    if len(grp) > 2:
        r2 = r2_score(grp['ET_pure_ICOS (mm/h)'], grp['ET_pixel (mm/h)_x'])
        rmse = np.sqrt(mean_squared_error(grp['ET_pure_ICOS (mm/h)'], grp['ET_pixel (mm/h)_x']))
        print(f"{s:15s}: R2 = {r2:6.3f} | RMSE = {rmse:5.3f} (N={len(grp)})")
