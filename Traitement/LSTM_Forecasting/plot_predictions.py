import os
import argparse
import torch
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import SITES_PILOTES, OUTPUT_DIR
from Traitement.LSTM_Forecasting.dataset_prep import build_continuous_dataset, create_sequences
from Traitement.LSTM_Forecasting.model_lstm import Seq2SeqLSTM
from Traitement.LSTM_Forecasting.train_lstm import get_device

LAMBDA_V = 2.45e6

def get_all_icos_le(site, dates_list):
    path = os.path.join(os.path.dirname(OUTPUT_DIR), "Outputs_ICOS", f"donnees_icos_LE_{site}.csv")
    if not os.path.exists(path):
        return [np.nan] * len(dates_list)
        
    df = pd.read_csv(path, index_col='TIMESTAMP', parse_dates=True)
    if 'LE_Consolide' not in df.columns:
        return [np.nan] * len(dates_list)
        
    results = []
    for date_obj in dates_list:
        date_str = date_obj.strftime("%Y-%m-%d")
        target_dt = pd.to_datetime(f"{date_str} 10:30:00")
        start = target_dt - pd.Timedelta(minutes=30)
        end = target_dt + pd.Timedelta(minutes=30)
        
        sub = df.loc[start:end]
        if sub.empty:
            results.append(np.nan)
        else:
            results.append(sub['LE_Consolide'].mean())
            
    return results

def plot_predictions():
    parser = argparse.ArgumentParser(description="Inférence du modèle LSTM")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                        help="Forcer l'exécution sur 'cpu' ou 'cuda'")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")
    
    out_dir = os.path.join(OUTPUT_DIR, "Modeles_ML")
    model_path = os.path.join(out_dir, "lstm_et_forecaster.pt")
    
    if not os.path.exists(model_path):
        print(f"Erreur: Modèle introuvable ({model_path})")
        return
        
    scaler_enc = joblib.load(os.path.join(out_dir, "scaler_enc.pkl"))
    scaler_dec = joblib.load(os.path.join(out_dir, "scaler_dec.pkl"))
    scaler_y = joblib.load(os.path.join(out_dir, "scaler_y.pkl"))
    
    # Init Model
    model = None
    
    global_y_true = []
    global_y_pred = []
    
    global_icos_true = []
    global_icos_pred = []
    global_icos_pt = []

    for site in SITES_PILOTES.keys():
        if site in ["Bondville", "Goodwin_Creek"]:
            continue
        print(f"\n==========================================")
        print(f"Évaluation sur le site : {site}")
        print(f"==========================================")
        
        df_val = build_continuous_dataset(site, num_points=1, start_year="2024", end_year="2024")
        if df_val.empty:
            print(f"Avertissement: Pas de données pour 2024 sur {site}.")
            continue
            
        X_enc, X_dec, Y_true = create_sequences(df_val, lookback=14, forecast=7)
        if len(X_enc) == 0:
            print(f"Avertissement: Séquences vides pour {site}.")
            continue
            
        N_test, L_enc, F_enc = X_enc.shape
        _, L_dec, F_dec = X_dec.shape
        
        if model is None:
            model = Seq2SeqLSTM(
                encoder_input_dim=F_enc,
                decoder_input_dim=F_dec,
                hidden_dim=64,
                output_dim=1,
                num_layers=2,
                dropout=0.3
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            
        X_enc_scaled = scaler_enc.transform(X_enc.reshape(-1, F_enc)).reshape(N_test, L_enc, F_enc)
        X_dec_scaled = scaler_dec.transform(X_dec.reshape(-1, F_dec)).reshape(N_test, L_dec, F_dec)
        
        with torch.no_grad():
            x_enc_tensor = torch.tensor(X_enc_scaled, dtype=torch.float32).to(device)
            x_dec_tensor = torch.tensor(X_dec_scaled, dtype=torch.float32).to(device)
            predictions_scaled = model(x_enc_tensor, x_dec_tensor).cpu().numpy()
            
        predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(N_test, L_dec, 1)
        
        out_dir_eval = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "LSTM_Evaluation", site)
        os.makedirs(out_dir_eval, exist_ok=True)
        
        # 1. Séries Temporelles
        indices = np.random.choice(len(X_enc), min(4, len(X_enc)), replace=False)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Séries Temporelles - {site}", fontsize=16)
        axes = axes.flatten()
        for i, idx in enumerate(indices):
            ax = axes[i]
            past_y = X_enc[idx, :, 3]
            true_y = Y_true[idx, :, 0]
            pred_y = predictions[idx, :, 0]
            time_past = range(-14, 0)
            time_future = range(0, 7)
            ax.plot(time_past, past_y, label='Passé (PT-SINRH observé)', color='blue', marker='o')
            ax.plot(time_future, true_y, label='Futur (PT-SINRH cible)', color='green', marker='o')
            ax.plot(time_future, pred_y, label='Futur (LSTM)', color='red', marker='x', linestyle='--')
            ax.set_title(f"Séquence #{idx}")
            ax.set_xlabel("Jours")
            ax.set_ylabel("ET (mm/jour)")
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(out_dir_eval, "1_Series_Temporelles.png"), dpi=300)
        plt.close()
        
        y_true_flat = Y_true.flatten()
        y_pred_flat = predictions.flatten()
        global_y_true.extend(y_true_flat)
        global_y_pred.extend(y_pred_flat)
        
        # 2. Scatter Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true_flat, y_pred_flat, alpha=0.5, color='purple', s=10)
        min_val = min(np.min(y_true_flat), np.min(y_pred_flat))
        max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Parfait')
        r2 = r2_score(y_true_flat, y_pred_flat)
        plt.title(f"Scatter Plot de Corrélation - {site}\nR² = {r2:.3f}")
        plt.xlabel("Vraie Valeur (PT-SINRH)")
        plt.ylabel("Prédiction (LSTM)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir_eval, "2_Scatter_Plot.png"), dpi=300)
        plt.close()
        
        # 3. Comparaison ICOS
        y_pred_day1 = predictions[:, 0, 0]
        target_indices = np.arange(14, 14 + len(y_pred_day1))
        dates_day1 = df_val['Date'].iloc[target_indices].tolist()
        icos_le_list = get_all_icos_le(site, dates_day1)
        
        icos_et_true = []
        for le in icos_le_list:
            if pd.isna(le):
                icos_et_true.append(np.nan)
            else:
                icos_et_true.append(le * 3600.0 * 24.0 / LAMBDA_V)
                
        df_eval = pd.DataFrame({
            'Date': dates_day1,
            'ET_PT_SINRH': Y_true[:, 0, 0],
            'ET_LSTM_Pred': y_pred_day1,
            'ET_Vrai_ICOS': icos_et_true
        })
        df_eval_clean = df_eval.dropna(subset=['ET_Vrai_ICOS']).copy()
        df_eval_clean = df_eval_clean[(df_eval_clean['ET_Vrai_ICOS'] >= -2.0) & (df_eval_clean['ET_Vrai_ICOS'] <= 12.0)]
        
        if len(df_eval_clean) > 5:
            global_icos_true.extend(df_eval_clean['ET_Vrai_ICOS'].tolist())
            global_icos_pred.extend(df_eval_clean['ET_LSTM_Pred'].tolist())
            global_icos_pt.extend(df_eval_clean['ET_PT_SINRH'].tolist())
            
            plt.figure(figsize=(8, 8))
            plt.scatter(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_LSTM_Pred'], alpha=0.6, color='red', label='LSTM vs ICOS', marker='x')
            plt.scatter(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_PT_SINRH'], alpha=0.6, color='blue', label='PT-SINRH vs ICOS', marker='o')
            min_v = min(df_eval_clean[['ET_Vrai_ICOS', 'ET_LSTM_Pred', 'ET_PT_SINRH']].min().min(), 0)
            max_v = max(df_eval_clean[['ET_Vrai_ICOS', 'ET_LSTM_Pred', 'ET_PT_SINRH']].max().max(), 6)
            plt.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='1:1 Parfait')
            
            r2_lstm = r2_score(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_LSTM_Pred'])
            rmse_lstm = np.sqrt(np.mean((df_eval_clean['ET_LSTM_Pred'] - df_eval_clean['ET_Vrai_ICOS'])**2))
            
            r2_pt = r2_score(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_PT_SINRH'])
            rmse_pt = np.sqrt(np.mean((df_eval_clean['ET_PT_SINRH'] - df_eval_clean['ET_Vrai_ICOS'])**2))
            
            title = f"Comparaison ICOS - {site}\nLSTM : R²={r2_lstm:.2f} | RMSE={rmse_lstm:.2f}\nPT-SINRH : R²={r2_pt:.2f} | RMSE={rmse_pt:.2f}"
            plt.title(title)
            plt.xlabel("Mesure ICOS (mm/jour)")
            plt.ylabel("Modèles (mm/jour)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(out_dir_eval, "3_Scatter_Plot_vs_ICOS.png"), dpi=300)
            plt.close()
            
            # 4. Histogramme des Résidus
            residus_lstm = df_eval_clean['ET_LSTM_Pred'] - df_eval_clean['ET_Vrai_ICOS']
            residus_pt = df_eval_clean['ET_PT_SINRH'] - df_eval_clean['ET_Vrai_ICOS']
            
            plt.figure(figsize=(8, 6))
            plt.hist(residus_lstm, bins=20, alpha=0.5, color='red', label='Résidus LSTM')
            plt.hist(residus_pt, bins=20, alpha=0.5, color='blue', label='Résidus PT-SINRH')
            plt.axvline(x=0, color='k', linestyle='--', lw=2)
            plt.title(f"Histogramme des Résidus vs ICOS - {site}")
            plt.xlabel("Erreur (Prédiction - ICOS) [mm/jour]")
            plt.ylabel("Fréquence")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(out_dir_eval, "4_Histogramme_Residus.png"), dpi=300)
            plt.close()

    print(f"\n==========================================")
    print(f"Génération des Graphiques Globaux")
    print(f"==========================================")
    
    out_dir_global = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "LSTM_Evaluation", "Global")
    os.makedirs(out_dir_global, exist_ok=True)
    
    if len(global_y_true) > 0:
        plt.figure(figsize=(8, 8))
        plt.scatter(global_y_true, global_y_pred, alpha=0.2, color='purple', s=5)
        min_val = min(np.min(global_y_true), np.min(global_y_pred))
        max_val = max(np.max(global_y_true), np.max(global_y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        r2_g = r2_score(global_y_true, global_y_pred)
        plt.title(f"GLOBAL Scatter Plot - Tous Sites\nR² = {r2_g:.3f}")
        plt.xlabel("PT-SINRH")
        plt.ylabel("LSTM")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir_global, "GLOBAL_Scatter_Plot.png"), dpi=300)
        plt.close()
        
    if len(global_icos_true) > 5:
        plt.figure(figsize=(8, 8))
        plt.scatter(global_icos_true, global_icos_pred, alpha=0.3, color='red', label='LSTM')
        plt.scatter(global_icos_true, global_icos_pt, alpha=0.3, color='blue', label='PT-SINRH')
        min_v = min(np.min(global_icos_true), min(np.min(global_icos_pred), np.min(global_icos_pt)))
        max_v = max(np.max(global_icos_true), max(np.max(global_icos_pred), np.max(global_icos_pt)))
        plt.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2)
        
        r2_lstm = r2_score(global_icos_true, global_icos_pred)
        r2_pt = r2_score(global_icos_true, global_icos_pt)
        plt.title(f"GLOBAL ICOS Comparaison\nLSTM: R²={r2_lstm:.2f} | PT-SINRH: R²={r2_pt:.2f}")
        plt.xlabel("ICOS")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir_global, "GLOBAL_Scatter_vs_ICOS.png"), dpi=300)
        plt.close()
        
        # 4. GLOBAL Histogramme des Résidus
        global_residus_lstm = np.array(global_icos_pred) - np.array(global_icos_true)
        global_residus_pt = np.array(global_icos_pt) - np.array(global_icos_true)
        
        plt.figure(figsize=(8, 6))
        plt.hist(global_residus_lstm, bins=50, alpha=0.5, color='red', label='Résidus LSTM')
        plt.hist(global_residus_pt, bins=50, alpha=0.5, color='blue', label='Résidus PT-SINRH')
        plt.axvline(x=0, color='k', linestyle='--', lw=2)
        plt.title("GLOBAL Histogramme des Résidus vs ICOS")
        plt.xlabel("Erreur (Prédiction - ICOS) [mm/jour]")
        plt.ylabel("Fréquence")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir_global, "GLOBAL_Histogramme_Residus.png"), dpi=300)
        plt.close()
        
    print("Terminé ! Graphiques sauvegardés dans Outputs/Analyses_Graphiques/LSTM_Evaluation/")

if __name__ == "__main__":
    plot_predictions()
