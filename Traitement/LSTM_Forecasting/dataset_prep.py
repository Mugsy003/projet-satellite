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
from config import OUTPUT_DIR, SITES_PILOTES
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
    
    # Calcul variance locale 3x3
    c1 = uniform_filter(ndvi, size=3, mode='reflect')
    c2 = uniform_filter(ndvi**2, size=3, mode='reflect')
    variance = c2 - c1**2
    
    # Exclure les pixels non végétaux en mettant leur variance à l'infini
    variance[~mask_veg] = np.inf
    # On met la variance du pixel central à l'infini pour ne pas le sélectionner en double
    variance[center_r, center_c] = np.inf
    
    # 1. On sélectionne un large pool des pixels les plus purs
    pool_size = max((num_points - 1) * 20, 400)
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

def build_continuous_dataset(site, num_points=15, start_year="2021", end_year="2024"):
    """
    Construit un dataset journalier continu pour un site, en exploitant num_points pixels
    spatiaux pour faire de la Data Augmentation (Pixels homogènes).
    """
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
            
    # Filtre uniquement les jours ayant NDVI ET SAVI
    valid_dates = {d: p for d, p in dict_dates.items() if 'ndvi' in p and 'savi' in p}
    
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
            with rasterio.open(paths['ndvi']) as src_ndvi, rasterio.open(paths['savi']) as src_savi:
                transform = src_ndvi.transform
                crs = src_ndvi.crs
                
                # Si le TIF actuel a un CRS différent du premier TIF, on reprojette
                if str(crs) != str(base_crs):
                    transformer_reproj = Transformer.from_crs(base_crs, crs, always_xy=True)
                else:
                    transformer_reproj = None

                ndvi_arr = src_ndvi.read(1)
                savi_arr = src_savi.read(1)
                
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
                            'SAVI': savi_arr[r, c]
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
        
        # IQR stat
        for col in ['NDVI', 'SAVI']:
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
    
    # 3. Fusionner ERA5 et SAT (On merge sur la date pour CHAQUE point)
    df_full = pd.merge(df_sat_clean, df_era_daily, on='Date', how='inner')
    df_full = df_full.set_index('Date').sort_index()
    df_full = df_full.loc[f"{start_year}-01-01":f"{end_year}-12-31"].copy()
    
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
    
    df_full = df_full.rename(columns={'Ta (°C)': 'Ta', 'Rn (W/m²)': 'Rn', 'RH (%)': 'RH'})
    
    # Clipping IQR pour éviter les explosions statistiques, on le fait globalement
    for col in ['Ta', 'Rn', 'RH', 'PT_SINRH_ET']:
        Q1 = df_full[col].quantile(0.25)
        Q3 = df_full[col].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df_full[col] = np.clip(df_full[col], lb, ub)
    
    return df_full.dropna()

def create_sequences(df, lookback=14, forecast=7, add_noise=False):
    """
    Crée les fenêtres glissantes d'Entraînement.
    Le groupement est fait par Point_ID pour ne jamais mélanger les pixels.
    Si add_noise=True, on injecte un bruit croissant sur les variables du décodeur
    (T_air, Rn, RH) pour simuler l'incertitude des prévisions météorologiques.
    """
    X_enc, X_dec, Y = [], [], []
    features_enc = ['NDVI', 'SAVI', 'PT_SINRH_ET', 'Ta', 'Rn', 'RH', 'DOY_sin', 'DOY_cos']
    features_dec = ['Ta', 'Rn', 'RH', 'DOY_sin', 'DOY_cos']
    
    # Profils d'incertitude (écart-type croissant avec l'horizon)
    sigma_Ta = np.linspace(0.5, 3.5, forecast) # erreur en °C
    sigma_Rn_pct = np.linspace(0.05, 0.25, forecast) # erreur en pourcentage (5% à 25%)
    sigma_RH = np.linspace(2.0, 15.0, forecast) # erreur en % d'humidité
    
    for pt_id, df_pt in df.groupby('Point_ID'):
        df_pt = df_pt.sort_values('Date')
        arr_enc = df_pt[features_enc].values
        arr_dec = df_pt[features_dec].values
        arr_y = df_pt[['PT_SINRH_ET']].values
        
        for i in range(len(df_pt) - lookback - forecast + 1):
            x_e = arr_enc[i : i + lookback]
            x_d = np.copy(arr_dec[i + lookback : i + lookback + forecast])
            y = arr_y[i + lookback : i + lookback + forecast]
            
            if add_noise:
                # Sauvegarde du signe de Rn pour éviter les inversions physiques
                rn_was_positive = x_d[:, 1] > 0
                
                # Ajout de bruit gaussien
                noise_Ta = np.random.normal(0, sigma_Ta)
                noise_Rn = np.random.normal(0, sigma_Rn_pct * np.abs(x_d[:, 1]))
                noise_RH = np.random.normal(0, sigma_RH)
                
                x_d[:, 0] += noise_Ta
                
                # Appliquer le bruit sur Rn et s'assurer qu'un Rn positif reste positif
                noisy_rn = x_d[:, 1] + noise_Rn
                x_d[:, 1] = np.where(rn_was_positive, np.maximum(0, noisy_rn), noisy_rn)
                
                # L'humidité relative doit strictement rester entre 0 et 100%
                x_d[:, 2] = np.clip(x_d[:, 2] + noise_RH, 0, 100)
            
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
