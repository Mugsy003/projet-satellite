import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import uniform_filter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import OUTPUT_DIR, SITES_PILOTES, LSTM_LOOKBACK, LSTM_FORECAST
from Traitement.PT_SINRH.algorithme_PT_SINRH import calculate_pt_sinrh_et

LAMBDA_V = 2.45e6

def extract_datetime_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})h(\d{2})", filename)
    if match:
        return pd.to_datetime(f"{match.group(1)} {match.group(2)}:{match.group(3)}:00")
    
    match2 = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match2:
        return pd.to_datetime(f"{match2.group(1)} 10:30:00")
    return None

def _calc_delta(T: np.ndarray) -> np.ndarray:
    return 4098.0 * (0.6108 * np.exp(17.27 * T / (T + 237.3))) / ((T + 237.3) ** 2)

def get_homogeneous_points(ndvi_path, center_r, center_c, num_points):
    """
    Applique le filtre DMS (variance locale) pour extraire les pixels les plus homogènes.
    Le point central ICOS (center_r, center_c) est toujours garanti d'être inclus.
    """
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)
        
    # Tolérance plus permissive pour accepter les sols nus agricoles / cultures coupées
    mask_veg = (ndvi > 0.1) & (ndvi <= 1.0)
    
    # Calcul variance locale 3x3 tolérant aux NaNs
    valid = ~np.isnan(ndvi)
    ndvi_filled = np.nan_to_num(ndvi, nan=0.0)
    
    # Utilisation de constant cval=0 pour éviter les effets de bord
    sum_ndvi = uniform_filter(ndvi_filled, size=3, mode='constant', cval=0.0) * 9
    sum_sq_ndvi = uniform_filter(ndvi_filled**2, size=3, mode='constant', cval=0.0) * 9
    count_valid = uniform_filter(valid.astype(float), size=3, mode='constant', cval=0.0) * 9
    count_valid = np.round(count_valid)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_ndvi = sum_ndvi / count_valid
        mean_sq_ndvi = sum_sq_ndvi / count_valid
        variance = mean_sq_ndvi - mean_ndvi**2
        
    variance[count_valid < 5] = np.inf # Exiger au moins 5 pixels valides dans le 3x3
    variance[np.isnan(variance)] = np.inf
    
    # Exclure les pixels non végétaux en mettant leur variance à l'infini
    variance[~mask_veg] = np.inf
    # On met la variance du pixel central à l'infini pour ne pas le sélectionner en double
    variance[center_r, center_c] = np.inf
    
    # 1. On sélectionne un large pool des pixels les plus purs
    # Tolérance de variance augmentée (pool plus grand)
    pool_size = max((num_points - 1) * 100, 2000)
    flat_indices = np.argsort(variance, axis=None)[:pool_size]
    
    valid_candidates = []
    for idx in flat_indices:
        r, c = np.unravel_index(idx, variance.shape)
        if variance[r, c] < np.inf:
            valid_candidates.append((r, c, ndvi[r, c]))
            
    # 2. On trie ce pool de pixels purs par leur valeur de NDVI
    valid_candidates.sort(key=lambda x: x[2])
    
    final_rows = [center_r]
    final_cols = [center_c]
    
    # 3. On pioche équitablement dans ce pool trié pour maximiser la diversité du NDVI
    if len(valid_candidates) > 0:
        n_to_select = min(num_points - 1, len(valid_candidates))
        selected_indices = np.linspace(0, len(valid_candidates) - 1, n_to_select, dtype=int)
        
        for idx in selected_indices:
            r, c, val = valid_candidates[idx]
            final_rows.append(r)
            final_cols.append(c)
            
    # Convertir ces indices (row, col) en coordonnées géographiques (X, Y)
    # pour pouvoir les projeter dynamiquement sur les autres images TIF
    with rasterio.open(ndvi_path) as src:
        transform = src.transform
        crs = src.crs
    
    geo_coords = []
    for r, c in zip(final_rows, final_cols):
        # transform * (col, row) donne (x, y)
        x_p, y_p = transform * (c, r)
        geo_coords.append((x_p, y_p))
        
    return geo_coords, crs

def build_continuous_dataset(site, num_points=20, start_date="2021-01-01", end_date="2024-12-31", include_openmeteo=False, use_cache=True):
    """
    Construit un dataset journalier continu pour un site, en exploitant num_points pixels
    spatiaux pour faire de la Data Augmentation (Pixels homogènes).
    Si use_cache=True, le dataset est sauvegardé/chargé depuis Outputs/Cache_Datasets/.
    """
    # --- Cache ---
    cache_dir = os.path.join(OUTPUT_DIR, "Cache_Datasets")
    os.makedirs(cache_dir, exist_ok=True)
    om_tag = "OM" if include_openmeteo else "ERA"
    cache_file = os.path.join(cache_dir, f"{site}_{start_date}_{end_date}_{num_points}pts_{om_tag}.parquet")
    
    if use_cache and os.path.exists(cache_file):
        print(f"[{site}] ⚡ Chargement depuis le cache ({os.path.basename(cache_file)})")
        return pd.read_parquet(cache_file)
    
    print(f"[{site}] Construction du dataset continu (Spatial Augmentation: {num_points} pts)...")
    coords = SITES_PILOTES[site]
    
    folders_to_check = [
        os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data"),
        os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}_S2", "3_Indices", "TIF_Data")
    ]
    
    dict_dates = {}
    for tif_folder in folders_to_check:
        if os.path.exists(tif_folder):
            for f in os.listdir(tif_folder):
                if not f.endswith('.tif'): continue
                dt = extract_datetime_from_filename(f)
                if not dt: continue
                
                date_str = dt.strftime("%Y-%m-%d")
                if date_str not in dict_dates: dict_dates[date_str] = {}
                if "NDVI" in f: dict_dates[date_str]['ndvi'] = os.path.join(tif_folder, f)
                if "SAVI" in f: dict_dates[date_str]['savi'] = os.path.join(tif_folder, f)
                if "NDWI" in f: dict_dates[date_str]['ndwi'] = os.path.join(tif_folder, f)
            
    # Filtre uniquement les jours ayant NDVI, SAVI et NDWI
    valid_dates = {d: p for d, p in dict_dates.items() if 'ndvi' in p and 'savi' in p and 'ndwi' in p}
    
    if not valid_dates:
        print(f"[{site}] Attention, aucune donnée satellite trouvée.")
        return pd.DataFrame()

    # Trouver les points de référence (utiliser la première image dispo)
    first_ndvi = list(valid_dates.values())[0]['ndvi']
    
    with rasterio.open(first_ndvi) as src:
        transform = src.transform
        crs = src.crs
        transformer_proj = Transformer.from_crs("EPSG:4326", str(crs), always_xy=True)
        x_p_center, y_p_center = transformer_proj.transform(coords["lon"], coords["lat"])
        col_idx = int((x_p_center - transform.c) / transform.a)
        row_idx = int((y_p_center - transform.f) / transform.e)
        
    geo_coords, base_crs = get_homogeneous_points(first_ndvi, row_idx, col_idx, num_points)
    print(f"   -> {len(geo_coords)} pixels valides trouvés pour l'extraction spatiale.")
    
    sparse_data = []
    for date_str, paths in valid_dates.items():
        try:
            with rasterio.open(paths['ndvi']) as src_ndvi, rasterio.open(paths['savi']) as src_savi, rasterio.open(paths['ndwi']) as src_ndwi:
                transform = src_ndvi.transform
                crs = src_ndvi.crs
                
                # Si le TIF actuel a un CRS différent du premier TIF, on reprojette
                if str(crs) != str(base_crs):
                    transformer_reproj = Transformer.from_crs(base_crs, crs, always_xy=True)
                else:
                    transformer_reproj = None

                ndvi_arr = src_ndvi.read(1)
                savi_arr = src_savi.read(1)
                ndwi_arr = src_ndwi.read(1)
                
                for pt_id, (x_p, y_p) in enumerate(geo_coords):
                    if transformer_reproj:
                        x_curr, y_curr = transformer_reproj.transform(x_p, y_p)
                    else:
                        x_curr, y_curr = x_p, y_p
                        
                    # Conversion coord géographique -> pixel de l'image courante
                    c = int((x_curr - transform.c) / transform.a)
                    r = int((y_curr - transform.f) / transform.e)
                    
                    if 0 <= r < ndvi_arr.shape[0] and 0 <= c < ndvi_arr.shape[1]:
                        sparse_data.append({
                            'Point_ID': pt_id,
                            'Date': pd.to_datetime(date_str),
                            'NDVI': ndvi_arr[r, c],
                            'SAVI': savi_arr[r, c],
                            'NDWI': ndwi_arr[r, c]
                        })
        except Exception as e:
            print(f"Erreur sur {date_str}: {e}")
            pass

    df_sat = pd.DataFrame(sparse_data)
    if df_sat.empty:
        return pd.DataFrame()
        
    # Filtrage et interpolation PAR POINT SPATIAL
    clean_dfs = []
    for pt_id, df_pt in df_sat.groupby('Point_ID'):
        df_pt = df_pt.copy()
        # Filtrage strict
        df_pt['NDVI'] = np.where((df_pt['NDVI'] < -1) | (df_pt['NDVI'] > 1), np.nan, df_pt['NDVI'])
        df_pt['SAVI'] = np.where((df_pt['SAVI'] < -1) | (df_pt['SAVI'] > 1), np.nan, df_pt['SAVI'])
        df_pt['NDWI'] = np.where((df_pt['NDWI'] < -1) | (df_pt['NDWI'] > 1), np.nan, df_pt['NDWI'])
        
        # IQR stat
        for col in ['NDVI', 'SAVI', 'NDWI']:
            Q1 = df_pt[col].quantile(0.25)
            Q3 = df_pt[col].quantile(0.75)
            IQR = Q3 - Q1
            lb = Q1 - 1.5 * IQR
            ub = Q3 + 1.5 * IQR
            df_pt[col] = np.where((df_pt[col] < lb) | (df_pt[col] > ub), np.nan, df_pt[col])
            
        df_pt = df_pt.dropna()
        df_pt = df_pt.groupby('Date').mean().reset_index()
        df_pt = df_pt.set_index('Date').resample('D').mean()
        df_pt = df_pt.interpolate(method='time').bfill().ffill()
        df_pt['Point_ID'] = pt_id
        clean_dfs.append(df_pt)
        
    df_sat_clean = pd.concat(clean_dfs)
    df_sat_clean = df_sat_clean.reset_index()

    # 2. Chargement de ERA5
    era5_path = os.path.join(os.path.dirname(OUTPUT_DIR), "Extraction", "ERA5", "Outputs_ERA5", f"donnees_era5_{site}.csv")
    if not os.path.exists(era5_path):
        era5_path = os.path.join(os.path.dirname(OUTPUT_DIR), "Outputs_ERA5", f"donnees_era5_{site}.csv")
        
    if not os.path.exists(era5_path):
        return pd.DataFrame()
        
    df_era = pd.read_csv(era5_path)
    df_era['TIMESTAMP'] = pd.to_datetime(df_era['TIMESTAMP'])
    df_era['Date'] = df_era['TIMESTAMP'].dt.normalize()
    
    df_era_daily = df_era.groupby('Date').agg({
        'Ta (°C)': 'mean', 'u (m/s)': 'mean', 'RH (%)': 'mean',
        'Rn (W/m²)': 'mean', 'R_s_down (W/m²)': 'mean', 'Pa (kPa)': 'mean'
    }).reset_index()
    
    # 2.5 Chargement de Open-Meteo (Prévisions)
    if include_openmeteo:
        openmeteo_path = os.path.join(OUTPUT_DIR, "Extraction", "OpenMeteo", f"donnees_openmeteo_{site}.csv")
        if not os.path.exists(openmeteo_path):
            print(f"Attention, fichier Open-Meteo introuvable pour {site}.")
            return pd.DataFrame()
            
        df_om = pd.read_csv(openmeteo_path)
        
        # Sécurité : Si le fichier a été téléchargé avant la mise à jour (pas de Rs), on l'ignore
        if 'Rs_fcst_J1' not in df_om.columns:
            print(f"Attention, la colonne Rs_fcst_J1 est manquante pour {site} (probablement bloqué par limite API 429). On ignore ce site pour l'instant.")
            return pd.DataFrame()
            
        df_om['TIMESTAMP'] = pd.to_datetime(df_om['TIMESTAMP'])
        df_om['Date'] = df_om['TIMESTAMP'].dt.normalize()
        
        agg_dict = {}
        for d in range(1, 8):
            agg_dict[f'Ta_fcst_J{d}'] = 'mean'
            agg_dict[f'RH_fcst_J{d}'] = 'mean'
            agg_dict[f'Rs_fcst_J{d}'] = 'mean'
            
        df_om_daily = df_om.groupby('Date').agg(agg_dict).reset_index()
    
    # 3. Fusionner ERA5, SAT et Open-Meteo (Inner join pour garantir la présence des prévisions si demandé)
    df_full = pd.merge(df_sat_clean, df_era_daily, on='Date', how='inner')
    if include_openmeteo:
        df_full = pd.merge(df_full, df_om_daily, on='Date', how='inner')
    df_full = df_full.set_index('Date').sort_index()
    df_full = df_full.loc[start_date:end_date].copy()
    
    # 4. Calcul dynamique PT-SINRH pour chaque jour et chaque point (Vecteur)
    T = len(df_full)
    if T == 0:
        return pd.DataFrame()
        
    ta = df_full['Ta (°C)'].values.astype(np.float32)
    rh = df_full['RH (%)'].values.astype(np.float32) / 100.0
    rn = df_full['Rn (W/m²)'].values.astype(np.float32)
    sw_in = df_full['R_s_down (W/m²)'].values.astype(np.float32)
    
    e_sat = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
    vpd = np.maximum(e_sat * (1.0 - rh), 0.01)
    g = np.where(rn > 0, rn * 0.1, 0).astype(np.float32)
    
    sw_in = np.where(np.isnan(sw_in) | (sw_in <= 0), np.maximum(rn * 1.2, 50.0), sw_in).astype(np.float32)
    
    delta = _calc_delta(ta).astype(np.float32)
    par = (sw_in * 0.48).astype(np.float32)
    
    ndvi_arr = df_full['NDVI'].values.astype(np.float32)
    savi_arr = df_full['SAVI'].values.astype(np.float32)
    
    res = calculate_pt_sinrh_et(
        RH=rh.reshape(T, 1, 1), Rn=rn.reshape(T, 1, 1), G=g.reshape(T, 1, 1),
        T_max=ta.reshape(T, 1, 1), NDVI=ndvi_arr.reshape(T, 1, 1), SAVI=savi_arr.reshape(T, 1, 1),
        Delta=delta.reshape(T, 1, 1), VPD=vpd.reshape(T, 1, 1), PAR=par.reshape(T, 1, 1)
    )
    
    et_mm_j = res['ET'][:, 0, 0] * 3600.0 * 24.0 / LAMBDA_V
    
    df_full['PT_SINRH_ET'] = et_mm_j
    df_full = df_full.reset_index()
    
    doy = df_full['Date'].dt.dayofyear
    df_full['DOY_sin'] = np.sin(2 * np.pi * doy / 365.25)
    df_full['DOY_cos'] = np.cos(2 * np.pi * doy / 365.25)
    df_full = df_full.rename(columns={'Ta (°C)': 'Ta', 'Rn (W/m²)': 'Rn', 'RH (%)': 'RH', 'R_s_down (W/m²)': 'Rs'})
    
    # Clipping IQR pour éviter les explosions statistiques, on le fait globalement
    for col in ['Ta', 'Rn', 'RH', 'PT_SINRH_ET']:
        Q1 = df_full[col].quantile(0.25)
        Q3 = df_full[col].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_full[col] = np.clip(df_full[col], lb, ub)
    
    result = df_full.dropna()
    
    # --- Sauvegarde en cache ---
    if use_cache and not result.empty:
        result.to_parquet(cache_file, index=False)
        print(f"[{site}] ✅ Dataset sauvegardé en cache ({os.path.basename(cache_file)}, {len(result)} lignes)")
    
    return result

def create_sequences(df, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST, use_true_forecast=True, add_noise=False):
    """
    Crée les fenêtres glissantes d'Entraînement.
    Le groupement est fait par Point_ID pour ne jamais mélanger les pixels.
    On utilise les données ERA5/SAT pour l'encodeur (passé).
    Le décodeur utilise les VRAIES prévisions J1 à J7 extraites le dernier jour du lookback.
    """
    X_enc, X_dec, Y = [], [], []
    features_enc = ['NDVI', 'SAVI', 'NDWI', 'PT_SINRH_ET', 'Ta', 'RH', 'Rn']
    
    for pt_id, df_pt in df.groupby('Point_ID'):
        df_pt = df_pt.sort_values('Date').reset_index(drop=True)
        arr_enc = df_pt[features_enc].values
        arr_y = df_pt[['PT_SINRH_ET']].values
        
        for i in range(len(df_pt) - lookback - forecast + 1):
            x_e = arr_enc[i : i + lookback]
            y = arr_y[i + lookback : i + lookback + forecast]
            
            x_d = np.zeros((forecast, 3), dtype=np.float32)
            
            if use_true_forecast:
                # Jour T = i + lookback - 1 (Le jour où la prévision est émise)
                row_T = df_pt.iloc[i + lookback - 1]
                for f_idx in range(forecast):
                    x_d[f_idx, 0] = row_T[f'Ta_fcst_J{f_idx+1}']
                    x_d[f_idx, 1] = row_T[f'RH_fcst_J{f_idx+1}']
                    x_d[f_idx, 2] = row_T[f'Rs_fcst_J{f_idx+1}']
            else:
                # Extraction des vraies valeurs futures (ERA5)
                future_Ta = arr_enc[i + lookback : i + lookback + forecast, 4] # Ta
                future_RH = arr_enc[i + lookback : i + lookback + forecast, 5] # RH
                future_Rs = df_pt['Rs'].values[i + lookback : i + lookback + forecast] # Rs (il n'est pas dans features_enc, on le prend dans df_pt)
                
                # Statistiques d'erreur empiriques d'Open-Meteo recalculées
                BIAS_TA = [-1.863, -1.841, -1.829, -1.804, -1.833, -1.840, -2.022]
                STD_TA  = [2.531, 3.398, 3.819, 4.062, 4.241, 4.337, 4.331]
                RHO_TA = 0.10
                
                BIAS_RH = [6.732, 5.609, 5.607, 5.512, 6.170, 6.332, 4.414]
                STD_RH  = [10.599, 11.984, 12.505, 12.751, 12.727, 12.788, 13.527]
                RHO_RH = 0.08
                
                # Nouveaux paramètres Rs empiriques (très forte sous-estimation par OM)
                BIAS_RS = [-223.155, -219.763, -219.316, -219.515, -219.591, -215.666, -195.300]
                STD_RS  = [154.365, 162.875, 164.515, 166.225, 168.069, 173.899, 167.296]
                RHO_RS = 0.40
                
                # Initialisation des processus AR(1) pour la trajectoire (N(0,1))
                x_ta = np.random.normal(0, 1) if add_noise else 0
                x_rh = np.random.normal(0, 1) if add_noise else 0
                x_rs = np.random.normal(0, 1) if add_noise else 0
                
                for f_idx in range(forecast):
                    if add_noise:
                        # Processus AR(1): X_t = rho * X_{t-1} + sqrt(1 - rho^2) * Z_t
                        if f_idx > 0:
                            x_ta = RHO_TA * x_ta + np.sqrt(1 - RHO_TA**2) * np.random.normal(0, 1)
                            x_rh = RHO_RH * x_rh + np.sqrt(1 - RHO_RH**2) * np.random.normal(0, 1)
                            x_rs = RHO_RS * x_rs + np.sqrt(1 - RHO_RS**2) * np.random.normal(0, 1)
                            
                        # Dénormalisation avec biais et std empiriques
                        ta_val = future_Ta[f_idx] + BIAS_TA[f_idx] + STD_TA[f_idx] * x_ta
                        rh_val = future_RH[f_idx] + BIAS_RH[f_idx] + STD_RH[f_idx] * x_rh
                        rs_val = future_Rs[f_idx] + BIAS_RS[f_idx] + STD_RS[f_idx] * x_rs
                        
                        # Cliping
                        rh_val = np.clip(rh_val, 10.0, 100.0)
                        rs_val = np.maximum(rs_val, 0.0)
                    else:
                        ta_val = future_Ta[f_idx]
                        rh_val = future_RH[f_idx]
                        rs_val = future_Rs[f_idx]
                        
                    x_d[f_idx, 0] = ta_val
                    x_d[f_idx, 1] = rh_val
                    x_d[f_idx, 2] = rs_val
                
                
            X_enc.append(x_e)
            X_dec.append(x_d)
            Y.append(y)
            
    return np.array(X_enc, dtype=np.float32), np.array(X_dec, dtype=np.float32), np.array(Y, dtype=np.float32)

if __name__ == "__main__":
    df = build_continuous_dataset("Gebesee", num_points=15)
    print(df.head())
    print(f"Total rows: {len(df)}")
    print(f"Unique points: {df['Point_ID'].nunique()}")
    if not df.empty:
        xe, xd, y = create_sequences(df)
        print("X_enc shape:", xe.shape)
