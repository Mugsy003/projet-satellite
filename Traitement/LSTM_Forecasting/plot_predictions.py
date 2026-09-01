import os
import argparse
import torch
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    SITES_PILOTES, OUTPUT_DIR, LSTM_LOOKBACK, LSTM_FORECAST,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT
)
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

def plot_predictions(model=None, device=None):
    if model is None or device is None:
        parser = argparse.ArgumentParser(description="Inférence du modèle LSTM")
        parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                            help="Forcer l'exécution sur 'cpu' ou 'cuda'")
        args = parser.parse_args()
        device = get_device(args.device)
        print(f"Device: {device}")
        
    out_dir = os.path.join(OUTPUT_DIR, "Modeles_ML")
    model_path = os.path.join(out_dir, "lstm_et_forecaster.pt")
    
    if model is None and not os.path.exists(model_path):
        print(f"Erreur: Modèle introuvable ({model_path})")
        return
        
    scaler_enc = joblib.load(os.path.join(out_dir, "scaler_enc.pkl"))
    scaler_dec = joblib.load(os.path.join(out_dir, "scaler_dec.pkl"))
    scaler_y = joblib.load(os.path.join(out_dir, "scaler_y.pkl"))
    
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
        
        # Validation sur 2025 pur
        df_val = build_continuous_dataset(site, num_points=50, start_date="2025-01-01", end_date="2025-12-31", include_openmeteo=True)
        if df_val.empty:
            print(f"Avertissement: Pas de données pour 2025 sur {site}.")
            continue
            
        X_enc, X_dec, Y_true = create_sequences(df_val, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST)
        if len(X_enc) == 0:
            print(f"Avertissement: Séquences vides pour {site}.")
            continue
            
        N_test, L_enc, F_enc = X_enc.shape
        _, L_dec, F_dec = X_dec.shape
        
        if model is None:
            model = Seq2SeqLSTM(
                encoder_input_dim=F_enc,
                decoder_input_dim=F_dec,
                hidden_dim=LSTM_HIDDEN_DIM,
                output_dim=1,
                num_layers=LSTM_NUM_LAYERS,
                dropout=LSTM_DROPOUT
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
            time_past = range(-LSTM_LOOKBACK, 0)
            time_future = range(0, LSTM_FORECAST)
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
        
        # 1b. Prévision Continue sur l'Année (Sauts de 7 jours)
        # On se limite au PREMIER PIXEL (Point_ID == 0) pour ne pas superposer 50 fois l'année !
        num_pixels = df_val['Point_ID'].nunique()
        seq_per_pixel = len(X_enc) // num_pixels
        df_val_pixel0 = df_val[df_val['Point_ID'] == df_val['Point_ID'].unique()[0]].reset_index(drop=True)
        
        jump_indices = np.arange(0, seq_per_pixel, LSTM_FORECAST)
        continuous_pred = []
        continuous_true = []
        continuous_dates = []
        
        for idx in jump_indices:
            continuous_pred.extend(predictions[idx, :, 0])
            continuous_true.extend(Y_true[idx, :, 0])
            
            # Extraire les dates depuis le dataframe du premier pixel
            target_indices = np.arange(LSTM_LOOKBACK + idx, LSTM_LOOKBACK + idx + LSTM_FORECAST)
            if target_indices[-1] < len(df_val_pixel0):
                continuous_dates.extend(df_val_pixel0['Date'].iloc[target_indices].tolist())
            else:
                diff = len(df_val_pixel0) - target_indices[0]
                if diff > 0:
                    continuous_dates.extend(df_val_pixel0['Date'].iloc[target_indices[:diff]].tolist())
                continuous_pred = continuous_pred[:len(continuous_dates)]
                continuous_true = continuous_true[:len(continuous_dates)]
                break

        mid_point = len(continuous_dates) // 2
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
        fig.suptitle(f"Simulation Complète 2025 - Sauts de 7 Jours - {site}", fontsize=16, fontweight='bold')
        
        # Première moitié
        ax1.plot(continuous_dates[:mid_point], continuous_true[:mid_point], label='Vérité Terrain (PT-SINRH)', color='green', linewidth=1.5)
        ax1.plot(continuous_dates[:mid_point], continuous_pred[:mid_point], label='Prédictions LSTM (Enchaînement)', color='red', linestyle='--', linewidth=1.5)
        ax1.set_title("Semestre 1", fontsize=13)
        ax1.set_ylabel("ET (mm/jour)")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Deuxième moitié
        ax2.plot(continuous_dates[mid_point:], continuous_true[mid_point:], label='Vérité Terrain (PT-SINRH)', color='green', linewidth=1.5)
        ax2.plot(continuous_dates[mid_point:], continuous_pred[mid_point:], label='Prédictions LSTM (Enchaînement)', color='red', linestyle='--', linewidth=1.5)
        ax2.set_title("Semestre 2", fontsize=13)
        ax2.set_ylabel("ET (mm/jour)")
        ax2.set_xlabel("Date")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(os.path.join(out_dir_eval, "1b_Prevision_Continue_Annuelle.png"), dpi=300)
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
        r_pearson, _ = pearsonr(y_true_flat, y_pred_flat)
        plt.title(f"Scatter Plot de Corrélation - {site}\nPearson R = {r_pearson:.3f}")
        plt.xlabel("Vraie Valeur (PT-SINRH)")
        plt.ylabel("Prédiction (LSTM)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir_eval, "2_Scatter_Plot.png"), dpi=300)
        plt.close()
        
        # 3. Comparaison ICOS
        y_pred_day1 = predictions[:seq_per_pixel, 0, 0]
        target_indices_px0 = np.arange(LSTM_LOOKBACK, LSTM_LOOKBACK + len(y_pred_day1))
        dates_day1 = df_val_pixel0['Date'].iloc[target_indices_px0].tolist()
        icos_le_list = get_all_icos_le(site, dates_day1)
        
        icos_et_true = []
        for le in icos_le_list:
            if pd.isna(le):
                icos_et_true.append(np.nan)
            else:
                icos_et_true.append(le * 3600.0 * 24.0 / LAMBDA_V)
                
        df_eval = pd.DataFrame({
            'Date': dates_day1,
            'ET_PT_SINRH': Y_true[:seq_per_pixel, 0, 0],
            'ET_LSTM_Pred': y_pred_day1,
            'ET_Vrai_ICOS': icos_et_true
        })
        df_eval_clean = df_eval.dropna(subset=['ET_Vrai_ICOS']).copy()
        df_eval_clean = df_eval_clean[(df_eval_clean['ET_Vrai_ICOS'] >= -2.0) & (df_eval_clean['ET_Vrai_ICOS'] <= 12.0)]
        
        if len(df_eval_clean) > 5:
            global_icos_true.extend(df_eval_clean['ET_Vrai_ICOS'].tolist())
            global_icos_pred.extend(df_eval_clean['ET_LSTM_Pred'].tolist())
            global_icos_pt.extend(df_eval_clean['ET_PT_SINRH'].tolist())
            
            r_lstm, _ = pearsonr(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_LSTM_Pred'])
            rmse_lstm = math.sqrt(mean_squared_error(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_LSTM_Pred']))
            rmse_pt = math.sqrt(mean_squared_error(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_PT_SINRH']))
            mae_pt = mean_absolute_error(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_PT_SINRH'])
            r_pt, _ = pearsonr(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_PT_SINRH'])
            
            plt.figure(figsize=(8, 8))
            plt.scatter(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_PT_SINRH'], alpha=0.5, label=f'PT-SINRH (R={r_pt:.2f})', color='gray')
            plt.scatter(df_eval_clean['ET_Vrai_ICOS'], df_eval_clean['ET_LSTM_Pred'], alpha=0.5, label=f'LSTM (R={r_lstm:.2f})', color='green')
            min_v = min(df_eval_clean[['ET_Vrai_ICOS', 'ET_LSTM_Pred', 'ET_PT_SINRH']].min().min(), 0)
            max_v = max(df_eval_clean[['ET_Vrai_ICOS', 'ET_LSTM_Pred', 'ET_PT_SINRH']].max().max(), 6)
            plt.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='1:1 Parfait')
            
            title = f"Comparaison ICOS - {site}\nLSTM : R={r_lstm:.2f} | RMSE={rmse_lstm:.2f}\nPT-SINRH : R={r_pt:.2f} | RMSE={rmse_pt:.2f}"
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

            # 5. Export des prévisions J1 à J7 et analyse par horizon
            out_dir_hor = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "LSTM_Evaluation", "Previsions_J_a_J7", site)
            os.makedirs(out_dir_hor, exist_ok=True)
            
            rmse_per_day = []
            
            for d in range(7):
                y_pred_d = predictions[:, d, 0]
                y_true_d = Y_true[:, d, 0]
                
                rmse_d = np.sqrt(np.mean((y_pred_d - y_true_d)**2))
                r_d, _ = pearsonr(y_true_d, y_pred_d)
                
                rmse_per_day.append(rmse_d)
                
                # Scatter pour ce jour
                plt.figure(figsize=(6, 6))
                plt.scatter(y_true_d, y_pred_d, alpha=0.3, color='orange', s=10)
                min_v = min(np.min(y_true_d), np.min(y_pred_d))
                max_v = max(np.max(y_true_d), np.max(y_pred_d))
                plt.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2)
                plt.title(f"Prévision à J+{d+1} - {site}\nR={r_d:.3f} | RMSE={rmse_d:.3f}")
                plt.xlabel("Vraie Valeur (PT-SINRH)")
                plt.ylabel(f"Prédiction LSTM (J+{d+1})")
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(out_dir_hor, f"Scatter_J_plus_{d+1}.png"), dpi=300)
                plt.close()
                
            # Graphe RMSE vs Horizon
            plt.figure(figsize=(8, 5))
            plt.plot(range(1, 8), rmse_per_day, marker='o', color='red', lw=2)
            plt.title(f"Dégradation de la RMSE selon l'horizon de prévision - {site}")
            plt.xlabel("Horizon de prévision (Jours)")
            plt.ylabel("RMSE (mm/jour)")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(out_dir_hor, "0_RMSE_vs_Horizon.png"), dpi=300)
            plt.close()
            
            # Export CSV des séquences
            records = []
            for n in range(N_test):
                for d in range(7):
                    target_idx = LSTM_LOOKBACK + n + d
                    if target_idx < len(df_val):
                        date_str = df_val['Date'].iloc[target_idx].strftime('%Y-%m-%d') if hasattr(df_val['Date'].iloc[target_idx], 'strftime') else df_val['Date'].iloc[target_idx]
                        records.append({
                            'Sequence_ID': n,
                            'Lead_Time_Days': d + 1,
                            'Date_Cible': date_str,
                            'ET_LSTM': predictions[n, d, 0],
                            'ET_PT_SINRH': Y_true[n, d, 0]
                        })
            df_export = pd.DataFrame(records)
            df_export.to_csv(os.path.join(out_dir_hor, f"{site}_Previsions_Detaillees.csv"), index=False, float_format="%.3f")

    print(f"\n==========================================")
    print(f"Génération des Graphiques Globaux")
    print(f"==========================================")
    
    out_dir_global = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "LSTM_Evaluation", "Global")
    os.makedirs(out_dir_global, exist_ok=True)
    
    if len(global_y_true) > 0:
        rmse_g = math.sqrt(mean_squared_error(global_y_true, global_y_pred))
        plt.figure(figsize=(8, 8))
        plt.scatter(global_y_true, global_y_pred, alpha=0.2, color='purple', s=5)
        min_val = min(np.min(global_y_true), np.min(global_y_pred))
        max_val = max(np.max(global_y_true), np.max(global_y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        r_g, _ = pearsonr(global_y_true, global_y_pred)
        plt.title(f"GLOBAL Scatter Plot - Tous Sites\nRMSE: {rmse_g:.2f} | R = {r_g:.3f}")
        plt.xlabel("PT-SINRH")
        plt.ylabel("LSTM")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(out_dir_global, "GLOBAL_Scatter_Plot.png"), dpi=300)
        plt.close()
        
    if len(global_icos_true) > 5:
        
        rmse_lstm = np.sqrt(np.mean((np.array(global_icos_pred) - np.array(global_icos_true))**2))
        mae_lstm = np.mean(np.abs(np.array(global_icos_pred) - np.array(global_icos_true)))
        bias_lstm = np.mean(np.array(global_icos_pred) - np.array(global_icos_true))

        rmse_pt = np.sqrt(np.mean((np.array(global_icos_pt) - np.array(global_icos_true))**2))
        mae_pt = np.mean(np.abs(np.array(global_icos_pt) - np.array(global_icos_true)))
        bias_pt = np.mean(np.array(global_icos_pt) - np.array(global_icos_true))

        # Recalcul des corrélations globales (et non celles du dernier site)
        r_lstm_global, _ = pearsonr(global_icos_true, global_icos_pred)
        r_pt_global, _ = pearsonr(global_icos_true, global_icos_pt)

        print(f"\n--- METRIQUES GLOBALES vs ICOS ---")
        print(f"LSTM     : R={r_lstm_global:.3f} | RMSE={rmse_lstm:.3f} | MAE={mae_lstm:.3f} | Biais={bias_lstm:.3f}")
        print(f"PT-SINRH : R={r_pt_global:.3f} | RMSE={rmse_pt:.3f} | MAE={mae_pt:.3f} | Biais={bias_pt:.3f}")

        plt.figure(figsize=(8, 8))
        min_v = min(np.min(global_icos_true), np.min(global_icos_pred), np.min(global_icos_pt), 0)
        max_v = max(np.max(global_icos_true), np.max(global_icos_pred), np.max(global_icos_pt), 6)
        plt.scatter(global_icos_true, global_icos_pt, alpha=0.3, label=f'PT-SINRH (R={r_pt_global:.2f})', color='gray', s=10)
        plt.scatter(global_icos_true, global_icos_pred, alpha=0.3, label=f'LSTM (R={r_lstm_global:.2f})', color='green', s=10)
        plt.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, label='1:1 Parfait')
        plt.title(f"GLOBAL ICOS Comparaison\nLSTM: R={r_lstm_global:.2f} | RMSE={rmse_lstm:.2f} | Biais={bias_lstm:.2f}\nPT-SINRH: R={r_pt_global:.2f} | RMSE={rmse_pt:.2f} | Biais={bias_pt:.2f}")
        plt.xlabel("Mesure ICOS (mm/jour)")
        plt.ylabel("Modèles (mm/jour)")
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
        
        # 5. GLOBAL Bar Chart Comparaison (R2, RMSE, MAE)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Comparaison Globale des Performances vs ICOS', fontsize=16)
        
        models = ['LSTM', 'PT-SINRH']
        colors = ['red', 'blue']
        
        # R (Pearson)
        bars1 = ax1.bar(models, [r_lstm_global, r_pt_global], color=colors, alpha=0.7)
        ax1.set_title('R (Pearson)')
        for bar in bars1:
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha='center', va='bottom', fontweight='bold')
            
        # RMSE
        bars2 = ax2.bar(models, [rmse_lstm, rmse_pt], color=colors, alpha=0.7)
        ax2.set_title('RMSE')
        for bar in bars2:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha='center', va='bottom', fontweight='bold')
            
        # MAE
        bars3 = ax3.bar(models, [mae_lstm, mae_pt], color=colors, alpha=0.7)
        ax3.set_title('MAE')
        for bar in bars3:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_global, "GLOBAL_Comparaison_Metriques.png"), dpi=300)
        plt.close()
        
    print("Terminé ! Graphiques sauvegardés dans Outputs/Analyses_Graphiques/LSTM_Evaluation/")

if __name__ == "__main__":
    plot_predictions()
