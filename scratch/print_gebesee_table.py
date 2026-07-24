import pandas as pd
df = pd.read_csv('Outputs/Toutes_Comparaisons/Comparaison_ET_Landsat_vs_PureICOS.csv')
geb = df[df['Site'] == 'Gebesee'].sort_values('Date')
print('| Date | Ta (°C) | u (m/s) | fc | LST_ICOS (°C) | T_s_max (°C) | T_c_max (°C) | ET_Pure (mm/h) |')
print('|------|---------|---------|----|---------------|--------------|--------------|----------------|')
cols = list(geb.columns)
lst_col = [c for c in cols if 'LST_ICOS' in c][0]
ta_col = [c for c in cols if 'Ta' in c][0]
ts_col = [c for c in cols if 'T_s_max' in c][0]
tc_col = [c for c in cols if 'T_c_max' in c][0]

for _, r in geb.iterrows():
    print(f"| {r['Date']} | {r[ta_col]:.1f} | {r['u (m/s)']:.1f} | {r['fc_pixel']:.2f} | {r[lst_col]:.1f} | {r[ts_col]:.1f} | {r[tc_col]:.1f} | {r['ET_pure_ICOS (mm/h)']:.3f} |")
