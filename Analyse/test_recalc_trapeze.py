import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SITE_TTME_PARAMS

RHO_CP = 1200.0
SIGMA = 5.67e-8
K_VK = 0.41
LAMBDA_V = 2.45e6
Z_M = 2.0
Z_H = 2.0

def compute_aerodynamic_resistance(u, z0m, z0h, z_m=Z_M):
    u = np.maximum(u, 0.5)
    r_ah = (np.log(z_m / z0m) * np.log(Z_H / z0h)) / (K_VK**2 * u)
    return r_ah

def main():
    # Load original ET TTME to get fc_pixel and Ta
    df = pd.read_csv("Outputs/Resultats_ET_TTME.csv")
    df_era5 = pd.read_csv("Outputs/Resultats_ET_TTME_ERA5.csv")
    
    et_pure_list = []
    
    for idx, row in df.iterrows():
        site = row['Site']
        date_str = row['Date']
        
        target_dt = pd.to_datetime(f"{date_str} 10:30:00")
        meteo_path = os.path.join("Outputs_ICOS", f"donnees_icos_{site}.csv")
        
        if not os.path.exists(meteo_path):
            et_pure_list.append(np.nan)
            continue
            
        df_meteo = pd.read_csv(meteo_path)
        df_meteo['TIMESTAMP'] = pd.to_datetime(df_meteo['TIMESTAMP'])
        diff = abs(df_meteo['TIMESTAMP'] - target_dt)
        mask = diff <= pd.Timedelta(minutes=60)
        
        if df_meteo[mask].empty:
            et_pure_list.append(np.nan)
            continue
            
        idx_best = diff[mask].idxmin()
        meteo_row = df_meteo.loc[idx_best]
        
        lst_ground = meteo_row.get('LST_Calculee', np.nan)
        sw_in = meteo_row.get('SW_IN_Consolide', np.nan)
        lw_in = meteo_row.get('LW_IN_Consolide', np.nan)
        Ta = meteo_row.get('TA_Consolide', np.nan)
        u = meteo_row.get('WS_Consolide', 2.0)
        u = max(u, 0.5)
        Rn = meteo_row.get('Rn_Consolide', np.nan)
        fc = row['fc_pixel']
        
        if pd.isna(lst_ground) or pd.isna(fc) or pd.isna(sw_in) or pd.isna(lw_in) or pd.isna(Ta) or pd.isna(Rn):
            et_pure_list.append(np.nan)
            continue
            
        params = SITE_TTME_PARAMS.get(site, SITE_TTME_PARAMS["default"])
        z0m_s = params.get("Z0M_SOIL", 0.005)
        z0h_s = params.get("Z0H_SOIL", 0.0005)
        z0m_c = params.get("Z0M_VEG", 0.10)
        z0h_c = params.get("Z0H_VEG", 0.01)
        cg_s  = params.get("C_G_SOIL", 0.30)
        cg_c  = params.get("C_G_VEG", 0.05)
        
        r_ah_s = compute_aerodynamic_resistance(u, z0m_s, z0h_s)
        r_ah_c = compute_aerodynamic_resistance(u, z0m_c, z0h_c)
        r_ah_s = min(r_ah_s, 110.0)
        r_ah_c = min(r_ah_c, 30.0)
        
        T_s_max_iter = Ta + 10.0
        for _ in range(3):
            Rn_s_theo = (1 - 0.25) * sw_in + 0.971 * lw_in - 0.971 * SIGMA * ((T_s_max_iter + 273.15)**4)
            T_s_max_iter = Ta + r_ah_s * (Rn_s_theo * (1.0 - cg_s)) / RHO_CP
        T_s_max = T_s_max_iter
        
        T_c_max_iter = Ta + 5.0
        for _ in range(3):
            Rn_c_theo = (1 - 0.20) * sw_in + 0.989 * lw_in - 0.989 * SIGMA * ((T_c_max_iter + 273.15)**4)
            T_c_max_iter = Ta + r_ah_c * (Rn_c_theo * (1.0 - cg_c)) / RHO_CP
        T_c_max = T_c_max_iter
        
        beta_w = T_c_max - T_s_max
        a = lst_ground - Ta
        T_warm_at_fc = T_s_max + beta_w * fc
        a_plus_b = T_warm_at_fc - Ta
        if a_plus_b > 0.1:
            ratio = np.clip(a / a_plus_b, 0.0, 1.0)
        else:
            ratio = np.nan
        beta_i = ratio * beta_w
        
        Ts = lst_ground - beta_i * fc
        Tc = Ts + beta_i
        Ts = max(Ts, Ta - 5)
        Tc = max(Tc, Ta - 5)
        
        G_pur_sol = cg_s * Rn
        G_pur_canopee = cg_c * Rn
        Rn_s_dispo = Rn - G_pur_sol
        Rn_c_dispo = Rn - G_pur_canopee
        
        if T_s_max > Ta:
            LE_s = Rn_s_dispo * (T_s_max - Ts) / (T_s_max - Ta)
        else:
            LE_s = 0.0
            
        if T_c_max > Ta:
            LE_c = Rn_c_dispo * (T_c_max - Tc) / (T_c_max - Ta)
        else:
            LE_c = 0.0
            
        LE = fc * LE_c + (1 - fc) * LE_s
        energie_dispo = Rn - (fc * G_pur_canopee + (1 - fc) * G_pur_sol)
        LE = np.clip(LE, 0.0, energie_dispo)
        
        ET = LE * 3600.0 / LAMBDA_V
        et_pure_list.append(ET)
        
    df['ET_pure_recalc'] = et_pure_list
    
    # Merge with ERA5 to compute R2
    df_merged = pd.merge(df, df_era5, on=['Site', 'Date'], suffixes=('', '_ERA5'))
    df_valid = df_merged.dropna(subset=['ET_pure_recalc', 'ET_pixel (mm/h)_ERA5'])
    
    print(f"Points valides : {len(df_valid)}")
    if len(df_valid) > 0:
        r2 = r2_score(df_valid['ET_pure_recalc'], df_valid['ET_pixel (mm/h)_ERA5'])
        print(f"R2 Global Pure ICOS (recalc) vs ERA5: {r2:.3f}")
        
        for site in df_valid['Site'].unique():
            df_s = df_valid[df_valid['Site'] == site]
            if len(df_s) >= 2:
                r2_s = r2_score(df_s['ET_pure_recalc'], df_s['ET_pixel (mm/h)_ERA5'])
                print(f"  {site}: R2 = {r2_s:.3f}")
    
if __name__ == '__main__':
    main()
