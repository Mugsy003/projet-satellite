import glob, os, json
sites = ['Gebesee', 'Selhausen', 'Lonzee', 'Voulundgaard', 'Klingenberg', 'Estrees-Mons', 'Borgo Cioffi', 'Grignon', 'Lamasquere']
res = {}
for s in sites:
    ds = set()
    for f in glob.glob(f'Outputs/Serie_Temporelle_{s}/3_Indices/TIF_Data/*_NDVI.tif'):
        ds.add(os.path.basename(f).split('_')[0])
    res[s] = list(ds)
with open('Outputs/manifest_dates_existantes.json', 'w') as f:
    json.dump(res, f)
print("Manifest updated.")
