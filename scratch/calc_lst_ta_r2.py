import pandas as pd
import glob
from sklearn.metrics import r2_score

sites = ['Gebesee', 'Selhausen', 'Lonzee', 'Voulundgaard', 'Klingenberg', 'Estrees-Mons', 'Borgo Cioffi', 'Grignon', 'Lamasquere']

dfs_icos = []
dfs_era5 = []

for s in sites:
    try:
        d1 = pd.read_csv(f'Outputs_ICOS/donnees_icos_{s}.csv')
        d1['Site'] = s
        dfs_icos.append(d1)
    except: pass
    
    try:
        d2 = pd.read_csv(f'Outputs_ERA5/donnees_era5_{s}.csv')
        d2['Site'] = s
        dfs_era5.append(d2)
    except: pass

df_icos = pd.concat(dfs_icos)
df_era5 = pd.concat(dfs_era5)

df_icos['TIMESTAMP'] = pd.to_datetime(df_icos['TIMESTAMP']).dt.tz_localize(None)
df_era5['TIMESTAMP'] = pd.to_datetime(df_era5.iloc[:, 0]).dt.tz_localize(None)

df = pd.merge(df_icos, df_era5, on=['Site', 'TIMESTAMP'], suffixes=('_ICOS', '_ERA5'))

df_clean_lst = df.dropna(subset=['LST_Calculee_ICOS', 'LST_Calculee_ERA5'])
r2_lst = r2_score(df_clean_lst['LST_Calculee_ICOS'], df_clean_lst['LST_Calculee_ERA5'])

# Also compute Ta
# ERA5 has 'TA_Consolide_ERA5' or 'Ta (°C)_ERA5'
ta_col_era5 = 'TA_Consolide_ERA5' if 'TA_Consolide_ERA5' in df.columns else 'Ta (°C)_ERA5'
df_clean_ta = df.dropna(subset=['TA_Consolide_ICOS', ta_col_era5])
r2_ta = r2_score(df_clean_ta['TA_Consolide_ICOS'], df_clean_ta[ta_col_era5])

print(f"R2 Native LST: {r2_lst:.4f}")
print(f"R2 Native Ta: {r2_ta:.4f}")
