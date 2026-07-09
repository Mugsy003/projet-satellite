import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

LAMBDA_V = 2.45e6

def main():
    df = pd.read_csv("Outputs/Resultats_ET_TTME.csv")
    df_era5 = pd.read_csv("Outputs/Resultats_ET_TTME_ERA5.csv")
    
    et_pure_list = []
    lst_pure_list = []
    
    for idx, row in df.iterrows():
        site = row['Site']
        date_str = row['Date']
        
        target_dt = pd.to_datetime(f"{date_str} 10:30:00")
        meteo_path = os.path.join("Outputs_ICOS", f"donnees_icos_{site}.csv")
        
        if not os.path.exists(meteo_path):
            et_pure_list.append(np.nan)
            lst_pure_list.append(np.nan)
            continue
            
        df_meteo = pd.read_csv(meteo_path)
        df_meteo['TIMESTAMP'] = pd.to_datetime(df_meteo['TIMESTAMP'])
        diff = abs(df_meteo['TIMESTAMP'] - target_dt)
        mask = diff <= pd.Timedelta(minutes=60)
        
        if df_meteo[mask].empty:
            et_pure_list.append(np.nan)
            lst_pure_list.append(np.nan)
            continue
            
        idx_best = diff[mask].idxmin()
        lst_ground = df_meteo.loc[idx_best, 'LST_Calculee']
        
        fc = row['fc_pixel']
        Ta = row['Ta (°C)']
        Rn = row['Rn (W/m²)']
        Ts_max = row['T_s_max (°C)']
        Tc_max = row['T_c_max (°C)']
        lst_landsat = row['LST_pixel (°C)']
        
        if pd.isna(lst_ground) or pd.isna(fc) or pd.isna(Ta) or pd.isna(Rn) or pd.isna(Ts_max) or pd.isna(lst_landsat):
            et_pure_list.append(np.nan)
            lst_pure_list.append(np.nan)
            continue
            
        # THE IDEA: Shift the trapezoid bounds by the exact bias between Ground LST and Landsat LST for this pixel
        # Or more realistically, we shift the Ground LST to the Landsat domain
        lst_bias = lst_ground - lst_landsat
        
        # Shift the trapezoid
        Ta_shifted = Ta + lst_bias
        Ts_max_shifted = Ts_max + lst_bias
        Tc_max_shifted = Tc_max + lst_bias
        
        beta_w = Tc_max_shifted - Ts_max_shifted
        a = lst_ground - Ta_shifted
        T_warm_at_fc = Ts_max_shifted + beta_w * fc
        a_plus_b = T_warm_at_fc - Ta_shifted
        
        if a_plus_b > 0.1:
            ratio = np.clip(a / a_plus_b, 0.0, 1.0)
        else:
            ratio = np.nan
            
        beta_i = ratio * beta_w
        Ts = lst_ground - beta_i * fc
        Tc = Ts + beta_i
        Ts = max(Ts, Ta_shifted - 5)
        Tc = max(Tc, Ta_shifted - 5)
        
        cg_s = 0.315 if fc < 0.5 else 0.35
        cg_c = 0.05
        G_pur_sol = cg_s * Rn
        G_pur_canopee = cg_c * Rn
        Rn_s_dispo = Rn - G_pur_sol
        Rn_c_dispo = Rn - G_pur_canopee
        
        if Ts_max_shifted > Ta_shifted:
            LE_s = Rn_s_dispo * (Ts_max_shifted - Ts) / (Ts_max_shifted - Ta_shifted)
        else:
            LE_s = 0.0
            
        if Tc_max_shifted > Ta_shifted:
            LE_c = Rn_c_dispo * (Tc_max_shifted - Tc) / (Tc_max_shifted - Ta_shifted)
        else:
            LE_c = 0.0
            
        LE = fc * LE_c + (1 - fc) * LE_s
        energie_dispo = Rn - (fc * G_pur_canopee + (1 - fc) * G_pur_sol)
        LE = np.clip(LE, 0.0, energie_dispo)
        
        ET = LE * 3600.0 / LAMBDA_V
        et_pure_list.append(ET)
        lst_pure_list.append(lst_ground)
        
    df['ET_pure_shifted'] = et_pure_list
    
    df_merged = pd.merge(df, df_era5, on=['Site', 'Date'], suffixes=('', '_ERA5'))
    df_valid = df_merged.dropna(subset=['ET_pure_shifted', 'ET_pixel (mm/h)_ERA5'])
    
    print(f"Points valides : {len(df_valid)}")
    if len(df_valid) > 0:
        r2 = r2_score(df_valid['ET_pure_shifted'], df_valid['ET_pixel (mm/h)_ERA5'])
        print(f"R2 Global Pure ICOS (shifted trapezoid) vs ERA5: {r2:.3f}")
        
        for site in df_valid['Site'].unique():
            df_s = df_valid[df_valid['Site'] == site]
            if len(df_s) >= 2:
                r2_s = r2_score(df_s['ET_pure_shifted'], df_s['ET_pixel (mm/h)_ERA5'])
                print(f"  {site}: R2 = {r2_s:.3f}")

if __name__ == '__main__':
    main()
