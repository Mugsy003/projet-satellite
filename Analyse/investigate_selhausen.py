import os
import pandas as pd
import numpy as np

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Outputs")
CSV_ERA5 = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons", "Comparaison_ET_ICOS_vs_ERA5.csv")
CSV_PURE = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons", "Comparaison_ET_Landsat_vs_PureICOS.csv")

def main():
    df_era5 = pd.read_csv(CSV_ERA5)
    df_pure = pd.read_csv(CSV_PURE)
    
    # Filtrer Selhausen
    df_era5 = df_era5[df_era5['Site'] == 'Selhausen']
    df_pure = df_pure[df_pure['Site'] == 'Selhausen']
    
    # Merge
    df = pd.merge(df_era5, df_pure[['Date', 'ET_pure_ICOS (mm/h)', 'LST_ICOS (°C)', 'LST_pixel (°C)']], on='Date')
    
    print("=" * 60)
    print("ANALYSE DU SITE : SELHAUSEN")
    print("=" * 60)
    print("Date       | LST(sat) | LST(sol) | Ta(ICOS) | Ta(ERA5) | u(ICOS) | u(ERA5) | Rn(ICOS) | Rn(ERA5) | ET(pure) | ET(ERA5) | ET(ICOS)")
    print("-" * 135)
    
    for idx, row in df.iterrows():
        d = row['Date']
        lst_sat = row['LST_pixel (°C)']
        lst_sol = row['LST_ICOS (°C)']
        ta_i = row['Ta (°C)_ICOS']
        ta_e = row['Ta (°C)_ERA5']
        u_i = row['u (m/s)_ICOS']
        u_e = row['u (m/s)_ERA5']
        rn_i = row['Rn (W/m²)_ICOS']
        rn_e = row['Rn (W/m²)_ERA5']
        
        et_p = row['ET_pure_ICOS (mm/h)']
        et_e = row['ET_pixel (mm/h)_ERA5']
        et_i = row['ET_pixel (mm/h)_ICOS']
        
        print(f"{d} | {lst_sat:>8.2f} | {lst_sol:>8.2f} | {ta_i:>8.2f} | {ta_e:>8.2f} | {u_i:>7.2f} | {u_e:>7.2f} | {rn_i:>8.1f} | {rn_e:>8.1f} | {et_p:>8.3f} | {et_e:>8.3f} | {et_i:>8.3f}")

if __name__ == "__main__":
    main()
