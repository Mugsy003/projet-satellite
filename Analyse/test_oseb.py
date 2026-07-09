import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SITE_TTME_PARAMS

RHO_CP = 1200.0
K_VK = 0.41
LAMBDA_V = 2.45e6
Z_M = 2.0
Z_H = 2.0

def main():
    df = pd.read_csv("Outputs/Resultats_ET_TTME.csv")
    df_era5 = pd.read_csv("Outputs/Resultats_ET_TTME_ERA5.csv")
    
    et_oseb_list = []
    
    for idx, row in df.iterrows():
        site = row['Site']
        date_str = row['Date']
        
        target_dt = pd.to_datetime(f"{date_str} 10:30:00")
        meteo_path = os.path.join("Outputs_ICOS", f"donnees_icos_{site}.csv")
        
        if not os.path.exists(meteo_path):
            et_oseb_list.append(np.nan)
            continue
            
        df_meteo = pd.read_csv(meteo_path)
        df_meteo['TIMESTAMP'] = pd.to_datetime(df_meteo['TIMESTAMP'])
        diff = abs(df_meteo['TIMESTAMP'] - target_dt)
        mask = diff <= pd.Timedelta(minutes=60)
        
        if df_meteo[mask].empty:
            et_oseb_list.append(np.nan)
            continue
            
        idx_best = diff[mask].idxmin()
        meteo_row = df_meteo.loc[idx_best]
        
        lst_ground = meteo_row.get('LST_Calculee', np.nan)
        Ta = meteo_row.get('TA_Consolide', np.nan)
        u = meteo_row.get('WS_Consolide', 2.0)
        u = max(u, 0.5)
        Rn = meteo_row.get('Rn_Consolide', np.nan)
        G = meteo_row.get('G_Consolide', np.nan)
        fc = row['fc_pixel']
        
        if pd.isna(lst_ground) or pd.isna(Ta) or pd.isna(Rn) or pd.isna(fc):
            et_oseb_list.append(np.nan)
            continue
            
        params = SITE_TTME_PARAMS.get(site, SITE_TTME_PARAMS["default"])
        z0m_s = params.get("Z0M_SOIL", 0.005)
        z0h_s = params.get("Z0H_SOIL", 0.0005)
        z0m_c = params.get("Z0M_VEG", 0.10)
        z0h_c = params.get("Z0H_VEG", 0.01)
        cg_s  = params.get("C_G_SOIL", 0.30)
        cg_c  = params.get("C_G_VEG", 0.05)
        
        # OSEB mathematical formula
        # 1. Effective roughness lengths
        z0m_eff = z0m_s * (1 - fc) + z0m_c * fc
        z0h_eff = z0h_s * (1 - fc) + z0h_c * fc
        
        # 2. Aerodynamic resistance
        r_ah = (np.log(Z_M / z0m_eff) * np.log(Z_H / z0h_eff)) / (K_VK**2 * u)
        r_ah = min(r_ah, 110.0)  # avoid extreme resistance
        
        # 3. Sensible Heat (H)
        H = RHO_CP * (lst_ground - Ta) / r_ah
        
        # 4. Soil Heat Flux (G)
        if pd.isna(G):
            G = (cg_s * (1 - fc) + cg_c * fc) * Rn
            
        # 5. Latent Heat (LE)
        LE = Rn - G - H
        
        # Physical constraints (LE cannot be negative or larger than available energy)
        energie_dispo = Rn - G
        LE = np.clip(LE, 0.0, max(energie_dispo, 0.0))
        
        # 6. ET
        ET = LE * 3600.0 / LAMBDA_V
        et_oseb_list.append(ET)
        
    df['ET_oseb'] = et_oseb_list
    
    df_merged = pd.merge(df, df_era5, on=['Site', 'Date'], suffixes=('', '_ERA5'))
    df_valid = df_merged.dropna(subset=['ET_oseb', 'ET_pixel (mm/h)_ERA5'])
    
    print(f"Points valides : {len(df_valid)}")
    if len(df_valid) > 0:
        r2 = r2_score(df_valid['ET_oseb'], df_valid['ET_pixel (mm/h)_ERA5'])
        print(f"R2 Global Math OSEB vs ERA5: {r2:.3f}")
        
        for site in df_valid['Site'].unique():
            df_s = df_valid[df_valid['Site'] == site]
            if len(df_s) >= 2:
                r2_s = r2_score(df_s['ET_oseb'], df_s['ET_pixel (mm/h)_ERA5'])
                print(f"  {site}: R2 = {r2_s:.3f}")

if __name__ == '__main__':
    main()
