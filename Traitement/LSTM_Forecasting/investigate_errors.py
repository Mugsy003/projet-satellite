import os
import torch
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import OUTPUT_DIR, LSTM_LOOKBACK, LSTM_FORECAST, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT
from Traitement.LSTM_Forecasting.dataset_prep import build_continuous_dataset, create_sequences
from Traitement.LSTM_Forecasting.model_lstm import Seq2SeqLSTM
from Traitement.LSTM_Forecasting.train_lstm import get_device

def investigate_errors():
    site = "Gebesee"
    print(f"Chargement des données 2024 pour {site}...")
    # Chargement du dataset continu
    df_val = build_continuous_dataset(site, num_points=50, start_date="2024-01-01", end_date="2024-12-31", include_openmeteo=True)
    
    if df_val.empty:
        print("Erreur: Pas de données pour 2024.")
        return
        
    X_enc, X_dec, Y_true = create_sequences(df_val, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST)
    
    device = get_device()
    model = Seq2SeqLSTM(
        encoder_input_dim=X_enc.shape[2],
        decoder_input_dim=X_dec.shape[2],
        hidden_dim=LSTM_HIDDEN_DIM,
        output_dim=1,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT
    ).to(device)
    
    out_dir = os.path.join(OUTPUT_DIR, "Modeles_ML")
    model_path = os.path.join(out_dir, "lstm_et_forecaster.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # ---------------------------------------------------------
    # APPLICATION DE LA NORMALISATION (SCALERS)
    # ---------------------------------------------------------
    scaler_enc = joblib.load(os.path.join(out_dir, "scaler_enc.pkl"))
    scaler_dec = joblib.load(os.path.join(out_dir, "scaler_dec.pkl"))
    scaler_y = joblib.load(os.path.join(out_dir, "scaler_y.pkl"))
    
    N_test, L_enc, F_enc = X_enc.shape
    _, L_dec, F_dec = X_dec.shape
    
    X_enc_scaled = scaler_enc.transform(X_enc.reshape(-1, F_enc)).reshape(N_test, L_enc, F_enc)
    X_dec_scaled = scaler_dec.transform(X_dec.reshape(-1, F_dec)).reshape(N_test, L_dec, F_dec)
    
    with torch.no_grad():
        x_enc_tensor = torch.tensor(X_enc_scaled, dtype=torch.float32).to(device)
        x_dec_tensor = torch.tensor(X_dec_scaled, dtype=torch.float32).to(device)
        predictions_scaled = model(x_enc_tensor, x_dec_tensor).cpu().numpy()
        
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(N_test, L_dec, 1)
        
    # On va extraire les prédictions pour le Jour J+1 (index 0 dans forecast)
    y_true_day1 = Y_true[:, 0, 0]
    y_pred_day1 = predictions[:, 0, 0]
    erreurs = y_pred_day1 - y_true_day1
    
    # Récupérer les Dates et Températures correspondantes
    # Pour la séquence i, le jour J+1 correspond à l'index df_val.index[i + lookback]
    target_indices = np.arange(LSTM_LOOKBACK, LSTM_LOOKBACK + len(y_true_day1))
    
    doys = df_val['Date'].iloc[target_indices].dt.dayofyear
    temperatures = df_val['Ta'].iloc[target_indices].values
    
    out_dir_eval = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "LSTM_Evaluation")
    os.makedirs(out_dir_eval, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Investigation des Erreurs (J+1) - {site} (2024)", fontsize=16)
    
    # Graphe 1: Erreur vs DOY
    ax1.scatter(doys, erreurs, alpha=0.6, color='red')
    ax1.axhline(y=0, color='black', linestyle='--')
    ax1.set_xlabel("Jour de l'année (DOY)")
    ax1.set_ylabel("Erreur (Pred - True) en mm/j")
    ax1.set_title("Distribution de l'Erreur selon la Saison")
    ax1.grid(True, alpha=0.3)
    
    # Graphe 2: Erreur vs Température
    ax2.scatter(temperatures, erreurs, alpha=0.6, color='darkorange')
    ax2.axhline(y=0, color='black', linestyle='--')
    ax2.set_xlabel("Température Moyenne (°C)")
    ax2.set_ylabel("Erreur (Pred - True) en mm/j")
    ax2.set_title("Distribution de l'Erreur selon la Température")
    ax2.grid(True, alpha=0.3)
    
    out_path = os.path.join(out_dir_eval, "5_Investigation_Erreurs.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"Graphique sauvegardé sous: {out_path}")

if __name__ == "__main__":
    investigate_errors()
