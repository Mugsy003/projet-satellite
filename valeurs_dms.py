import rioxarray
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
PATH_TIF = r"C:\Users\a951444\Workspace\projet-satellite\Outputs\Serie_Temporelle_Lamasquere\3_Indices\TIF_Data\2024-08-11_10h35_Lamasquere_Thermique_B10.tif"
OUTPUT_CSV = "valeurs_temperatures_B10.csv"

def extract_tif_to_csv(path_in, path_out):
    print(f"📖 Ouverture de {path_in}...")
    # Charger le TIF
    rds = rioxarray.open_rasterio(path_in)
    
    # Extraire les valeurs dans un tableau 1D (on ignore les NaNs/NoData)
    # .values[0] car c'est souvent une seule bande (LST)
    data = rds.values[0].flatten()
    
    # Nettoyage des NoData (souvent des valeurs très petites ou très grandes)
    nodata = rds.rio.nodata
    if nodata is not None:
        data = data[data != nodata]
    
    # Filtrage des valeurs aberrantes (0 ou inf)
    data = data[~np.isnan(data)]
    
    # Conversion Kelvin -> Celsius si les valeurs sont > 200
    if np.nanmean(data) > 200:
        print("🌡️ Détection de Kelvins, conversion en Celsius...")
        data = data - 273.15

    # Création du DataFrame
    df = pd.DataFrame(data, columns=['LST_Celsius'])
    
    # Sauvegarde
    df.to_csv(path_out, index=False)
    print(f"✅ Terminé ! {len(df)} pixels enregistrés dans {path_out}")

if __name__ == "__main__":
    extract_tif_to_csv(PATH_TIF, OUTPUT_CSV)