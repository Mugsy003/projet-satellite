import os
import argparse
import glob
import re
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import joblib
from scipy.interpolate import interp1d
import sys

# Ajouter le chemin racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import OUTPUT_DIR, SITES_PILOTES, LSTM_LOOKBACK, LSTM_FORECAST, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT
from Traitement.PT_SINRH.algorithme_PT_SINRH import calculate_pt_sinrh_et
from Traitement.LSTM_Forecasting.model_lstm import Seq2SeqLSTM
from Traitement.LSTM_Forecasting.train_lstm import get_device

LAMBDA_V = 2.45e6

def extract_date_from_filename(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return pd.to_datetime(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=str, default="Gebesee")
    parser.add_argument('--target_date', type=str, default="2025-06-15", help="Date de début de la prévision (J+1)")
    parser.add_argument('--window_size', type=int, default=150, help="Taille de la fenêtre en pixels (côté du carré)")
    parser.add_argument('--device', type=str, default="auto")
    args = parser.parse_args()

    site = args.site
    target_date = pd.to_datetime(args.target_date)
    window_size = args.window_size
    device = get_device(args.device)
    
    print(f"🌍 Lancement de la cartographie prédictive LSTM sur {site}")
    print(f"📅 Date de prévision cible : {target_date.strftime('%Y-%m-%d')} (J+1)")
    
    # 1. Définir les périodes
    forecast = LSTM_FORECAST
    lookback = LSTM_LOOKBACK
    start_history = target_date - pd.Timedelta(days=lookback)
    end_history = target_date - pd.Timedelta(days=1)
    end_forecast = target_date + pd.Timedelta(days=forecast - 1)
    
    # Dates continues
    history_dates = pd.date_range(start=start_history, end=end_history, freq='D')
    forecast_dates = pd.date_range(start=target_date, end=end_forecast, freq='D')
    
    # 2. Chercher les images satellites (jusqu'à 45 jours avant pour l'interpolation)
    tif_folder = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data")
    if not os.path.exists(tif_folder):
        tif_folder = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}_S2", "3_Indices", "TIF_Data")
        
    all_tifs = glob.glob(os.path.join(tif_folder, "*.tif"))
    dict_dates = {}
    for f in all_tifs:
        dt = extract_date_from_filename(f)
        if dt is None: continue
        if dt >= target_date - pd.Timedelta(days=45) and dt <= end_history:
            if dt not in dict_dates: dict_dates[dt] = {}
            if "NDVI" in f: dict_dates[dt]['ndvi'] = f
            if "SAVI" in f: dict_dates[dt]['savi'] = f
            if "NDWI" in f: dict_dates[dt]['ndwi'] = f
            
    valid_dates = sorted([d for d, p in dict_dates.items() if 'ndvi' in p and 'savi' in p and 'ndwi' in p])
    if len(valid_dates) < 2:
        print("Erreur: Pas assez d'images satellites pour interpoler l'historique.")
        return
        
    print(f"📡 {len(valid_dates)} dates satellites trouvées pour l'interpolation temporelle.")
    
    # Trouver le pixel ICOS
    coords = SITES_PILOTES[site]
    with rasterio.open(dict_dates[valid_dates[0]]['ndvi']) as src:
        transform_full = src.transform
        crs = src.crs
        transformer = Transformer.from_crs("EPSG:4326", str(crs), always_xy=True)
        x_icos, y_icos = transformer.transform(coords["lon"], coords["lat"])
        c_icos = int((x_icos - transform_full.c) / transform_full.a)
        r_icos = int((y_icos - transform_full.f) / transform_full.e)
        
    # Définir la fenêtre de rognage (crop)
    r_start = max(0, r_icos - window_size // 2)
    c_start = max(0, c_icos - window_size // 2)
    w_crop = window_size
    h_crop = window_size
    
    with rasterio.open(dict_dates[valid_dates[0]]['ndvi']) as src:
        r_start = min(r_start, src.height - h_crop)
        c_start = min(c_start, src.width - w_crop)
        window = Window(c_start, r_start, w_crop, h_crop)
        transform_crop = src.window_transform(window)
        
    # 3. Charger et interpoler les images 2D
    print("⏳ Interpolation temporelle des pixels 2D (NDVI, SAVI, NDWI)...")
    raw_times = np.array([(d - start_history).days for d in valid_dates])
    target_times = np.arange(lookback) # 0 to 13
    
    # Tableaux pour stocker les données brutes
    ndvi_raw = np.zeros((len(valid_dates), h_crop, w_crop), dtype=np.float32)
    savi_raw = np.zeros((len(valid_dates), h_crop, w_crop), dtype=np.float32)
    ndwi_raw = np.zeros((len(valid_dates), h_crop, w_crop), dtype=np.float32)
    
    for i, d in enumerate(valid_dates):
        with rasterio.open(dict_dates[d]['ndvi']) as src: ndvi_raw[i] = src.read(1, window=window)
        with rasterio.open(dict_dates[d]['savi']) as src: savi_raw[i] = src.read(1, window=window)
        with rasterio.open(dict_dates[d]['ndwi']) as src: ndwi_raw[i] = src.read(1, window=window)
        
    # Masque végétatif global
    mask_veg = (np.nanmean(ndvi_raw, axis=0) > 0.05)
    
    # 3.5 Remplissage des NaNs (Nuages historiques) avant l'interpolation
    # Si un pixel était un nuage dans le passé, interp1d propageait le NaN à toute la série.
    def fill_nans_3d(arr):
        arr_filled = arr.copy()
        for r in range(arr.shape[1]):
            for c in range(arr.shape[2]):
                pixel = arr[:, r, c]
                valid = ~np.isnan(pixel)
                if valid.any() and not valid.all():
                    pixel[~valid] = np.interp(np.nonzero(~valid)[0], np.nonzero(valid)[0], pixel[valid])
                arr_filled[:, r, c] = pixel
        return arr_filled

    ndvi_raw = fill_nans_3d(ndvi_raw)
    savi_raw = fill_nans_3d(savi_raw)
    ndwi_raw = fill_nans_3d(ndwi_raw)

    # Interpolation temporelle le long de l'axe 0
    interpolator_ndvi = interp1d(raw_times, ndvi_raw, axis=0, kind='linear', fill_value="extrapolate")
    interpolator_savi = interp1d(raw_times, savi_raw, axis=0, kind='linear', fill_value="extrapolate")
    interpolator_ndwi = interp1d(raw_times, ndwi_raw, axis=0, kind='linear', fill_value="extrapolate")
    
    ndvi_interp = interpolator_ndvi(target_times) # (14, h, w)
    savi_interp = interpolator_savi(target_times)
    ndwi_interp = interpolator_ndwi(target_times)
    
    # Limiter les valeurs aberrantes
    ndvi_interp = np.clip(ndvi_interp, -1, 1)
    savi_interp = np.clip(savi_interp, -1, 1)
    ndwi_interp = np.clip(ndwi_interp, -1, 1)
    
    # 4. Charger ERA5 pour calculer PT-SINRH
    print("⛅ Calcul du modèle physique PT-SINRH pour l'historique...")
    era5_path = os.path.join(os.path.dirname(OUTPUT_DIR), "Extraction", "ERA5", "Outputs_ERA5", f"donnees_era5_{site}.csv")
    if not os.path.exists(era5_path):
        era5_path = os.path.join(os.path.dirname(OUTPUT_DIR), "Outputs_ERA5", f"donnees_era5_{site}.csv")
    df_era = pd.read_csv(era5_path)
    df_era['TIMESTAMP'] = pd.to_datetime(df_era['TIMESTAMP'])
    df_era['Date'] = df_era['TIMESTAMP'].dt.normalize()
    df_era_daily = df_era.groupby('Date').mean(numeric_only=True).reset_index()
    
    df_era_hist = df_era_daily[(df_era_daily['Date'] >= start_history) & (df_era_daily['Date'] <= end_history)].copy()
    if len(df_era_hist) != lookback:
        print("Erreur: Données ERA5 incomplètes pour l'historique.")
        return
        
    ta_hist = df_era_hist['Ta (°C)'].values.astype(np.float32)
    rh_hist = df_era_hist['RH (%)'].values.astype(np.float32) / 100.0
    rn_hist = df_era_hist['Rn (W/m²)'].values.astype(np.float32)
    sw_hist = df_era_hist['R_s_down (W/m²)'].values.astype(np.float32)
    
    e_sat = 0.6108 * np.exp(17.27 * ta_hist / (ta_hist + 237.3))
    vpd = np.maximum(e_sat * (1.0 - rh_hist), 0.01)
    g = np.where(rn_hist > 0, rn_hist * 0.1, 0).astype(np.float32)
    sw_hist = np.where(np.isnan(sw_hist) | (sw_hist <= 0), np.maximum(rn_hist * 1.2, 50.0), sw_hist).astype(np.float32)
    
    def _calc_delta(T): return 4098.0 * (0.6108 * np.exp(17.27 * T / (T + 237.3))) / ((T + 237.3) ** 2)
    delta = _calc_delta(ta_hist).astype(np.float32)
    par = (sw_hist * 0.48).astype(np.float32)
    
    res_pt = calculate_pt_sinrh_et(
        RH=rh_hist.reshape(lookback, 1, 1), Rn=rn_hist.reshape(lookback, 1, 1), G=g.reshape(lookback, 1, 1),
        T_max=ta_hist.reshape(lookback, 1, 1), NDVI=ndvi_interp, SAVI=savi_interp,
        Delta=delta.reshape(lookback, 1, 1), VPD=vpd.reshape(lookback, 1, 1), PAR=par.reshape(lookback, 1, 1)
    )
    pt_et = res_pt['ET'] * 3600.0 * 24.0 / LAMBDA_V # (14, h, w)
    
    # 5. Charger Open-Meteo pour le prévisionnel
    # Le CSV Open-Meteo contient des colonnes : Ta_fcst_J1..J7, RH_fcst_J1..J7, Rs_fcst_J1..J7
    # On extrait les prévisions émises le dernier jour du lookback (end_history)
    openmeteo_path = os.path.join(OUTPUT_DIR, "Extraction", "OpenMeteo", f"donnees_openmeteo_{site}.csv")
    df_om = pd.read_csv(openmeteo_path)
    df_om['TIMESTAMP'] = pd.to_datetime(df_om['TIMESTAMP'])
    df_om['Date'] = df_om['TIMESTAMP'].dt.normalize()
    df_om_daily = df_om.groupby('Date').mean(numeric_only=True).reset_index()
    
    # On prend les prévisions émises le dernier jour du lookback
    df_om_anchor = df_om_daily[df_om_daily['Date'] == end_history]
    if df_om_anchor.empty:
        print(f"Erreur: Pas de prévisions Open-Meteo pour la date {end_history.strftime('%Y-%m-%d')}.")
        return
    row_anchor = df_om_anchor.iloc[0]
    
    ta_fcst = np.array([row_anchor[f'Ta_fcst_J{d}'] for d in range(1, forecast + 1)], dtype=np.float32)
    rh_fcst = np.array([row_anchor[f'RH_fcst_J{d}'] for d in range(1, forecast + 1)], dtype=np.float32)
    rs_fcst = np.array([row_anchor[f'Rs_fcst_J{d}'] for d in range(1, forecast + 1)], dtype=np.float32)
    
    # 6. Préparation des tenseurs pour le LSTM
    print("🧠 Préparation des tenseurs et inférence LSTM...")
    N_pixels = h_crop * w_crop
    
    ta_hist_3d = np.broadcast_to(ta_hist[:, np.newaxis, np.newaxis], (lookback, h_crop, w_crop))
    rh_hist_3d = np.broadcast_to(rh_hist[:, np.newaxis, np.newaxis] * 100.0, (lookback, h_crop, w_crop))
    rn_hist_3d = np.broadcast_to(rn_hist[:, np.newaxis, np.newaxis], (lookback, h_crop, w_crop))
    
    X_enc = np.stack([ndvi_interp, savi_interp, ndwi_interp, pt_et, ta_hist_3d, rh_hist_3d, rn_hist_3d], axis=-1)
    X_enc = X_enc.transpose(1, 2, 0, 3).reshape(N_pixels, lookback, 7)
    
    ta_fcst_3d = np.broadcast_to(ta_fcst[:, np.newaxis, np.newaxis], (forecast, h_crop, w_crop))
    rh_fcst_3d = np.broadcast_to(rh_fcst[:, np.newaxis, np.newaxis], (forecast, h_crop, w_crop))
    rs_fcst_3d = np.broadcast_to(rs_fcst[:, np.newaxis, np.newaxis], (forecast, h_crop, w_crop))
    
    X_dec = np.stack([ta_fcst_3d, rh_fcst_3d, rs_fcst_3d], axis=-1)
    X_dec = X_dec.transpose(1, 2, 0, 3).reshape(N_pixels, forecast, 3)
    
    model_dir = os.path.join(OUTPUT_DIR, "Modeles_ML")
    scaler_enc = joblib.load(os.path.join(model_dir, "scaler_enc.pkl"))
    scaler_dec = joblib.load(os.path.join(model_dir, "scaler_dec.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))
    
    X_enc_scaled = scaler_enc.transform(X_enc.reshape(-1, 7)).reshape(N_pixels, lookback, 7)
    X_dec_scaled = scaler_dec.transform(X_dec.reshape(-1, 3)).reshape(N_pixels, forecast, 3)
    
    model = Seq2SeqLSTM(encoder_input_dim=7, decoder_input_dim=3, hidden_dim=LSTM_HIDDEN_DIM, output_dim=1, num_layers=LSTM_NUM_LAYERS, dropout=LSTM_DROPOUT).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "lstm_et_forecaster.pt"), map_location=device, weights_only=True))
    model.eval()
    
    batch_size = 2048
    predictions_scaled = np.zeros((N_pixels, forecast, 1), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(0, N_pixels, batch_size):
            x_e = torch.tensor(X_enc_scaled[i:i+batch_size], dtype=torch.float32).to(device)
            x_d = torch.tensor(X_dec_scaled[i:i+batch_size], dtype=torch.float32).to(device)
            out = model(x_e, x_d)
            predictions_scaled[i:i+batch_size] = out.cpu().numpy()
            
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(N_pixels, forecast)
    predictions_map = predictions.reshape(h_crop, w_crop, forecast)
    
    predictions_map[~mask_veg] = np.nan
    
    # 7. Sauvegarde des images 2D
    out_dir_maps = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "Cartes_Spatiales", "ET_Predite_LSTM", site)
    os.makedirs(out_dir_maps, exist_ok=True)
    
    print("🗺️ Génération des cartes géospatiales...")
    c_icos_crop = c_icos - c_start
    r_icos_crop = r_icos - r_start
    
    for d in range(forecast):
        forecast_dt_str = forecast_dates[d].strftime('%Y-%m-%d')
        et_map_d = predictions_map[:, :, d]
        
        tif_path = os.path.join(out_dir_maps, f"{site}_{forecast_dt_str}_LSTM_J{d+1}.tif")
        with rasterio.open(
            tif_path, 'w', driver='GTiff',
            height=h_crop, width=w_crop, count=1,
            dtype=np.float32, crs=crs,
            transform=transform_crop, nodata=np.nan
        ) as dst:
            dst.write(et_map_d, 1)
            
        plt.figure(figsize=(10, 8))
        im = plt.imshow(et_map_d, cmap='YlGnBu', vmin=0, vmax=10.0)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="Évapotranspiration LSTM (mm/jour)")
        
        rect = patches.Rectangle((c_icos_crop-1.5, r_icos_crop-1.5), 3, 3, linewidth=2, edgecolor='red', facecolor='none', label='Tour ICOS')
        plt.gca().add_patch(rect)
        
        plt.title(f"Prévision LSTM J+{d+1} - {site} ({forecast_dt_str})", fontsize=14, pad=15)
        plt.legend(loc='upper right')
        
        png_path = os.path.join(out_dir_maps, f"{site}_{forecast_dt_str}_LSTM_J{d+1}.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"✅ Terminé ! {forecast} cartes sauvegardées dans {out_dir_maps}")

if __name__ == "__main__":
    main()
