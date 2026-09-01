import os
import argparse
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    OUTPUT_DIR, SITES_PILOTES, LSTM_LOOKBACK, LSTM_FORECAST,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    LSTM_EPOCHS, LSTM_LR, LSTM_WEIGHT_DECAY
)
from Traitement.LSTM_Forecasting.dataset_prep import build_continuous_dataset, create_sequences
from Traitement.LSTM_Forecasting.model_lstm import Seq2SeqLSTM

def get_device(device_arg=None):
    if device_arg == 'cuda' and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_arg == 'cpu':
        return torch.device("cpu")
        
    # Auto-détection si rien n'est forcé ou si cuda demandé mais non dispo
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class ShapeLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(ShapeLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        # 1. Erreur absolue
        huber_loss = self.huber(y_pred, y_true)
        
        # 2. Erreur sur la forme (Pénalité sur les variations temporelles)
        diff_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        diff_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        
        # Erreur quadratique standard sur les dérivées
        mse_shape = self.mse(diff_pred, diff_true)
        
        # 3. Pénalité d'anti-lissage (Anti-Over-Smoothing)
        # Force le modèle à varier au moins autant que la vérité
        # Si la variation prédite est inférieure à la variation réelle, on applique une forte pénalité
        under_variation = torch.relu(torch.abs(diff_true) - torch.abs(diff_pred))
        anti_smooth_loss = torch.mean(under_variation)
        
        # 4. Pénalité de variation minimale absolue (le modèle ne doit jamais être 100% plat)
        # On impose une variation minimale de 0.05 (en données normalisées)
        min_var_loss = torch.relu(0.05 - torch.abs(diff_pred)).mean()
        
        return huber_loss + self.alpha * mse_shape + 0.5 * anti_smooth_loss + 0.1 * min_var_loss

def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001, device='cpu'):
    criterion = ShapeLoss(alpha=0.6)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []
    from sklearn.metrics import r2_score
    
    # Early Stopping variables
    patience = 5
    epochs_no_improve = 0
    
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        all_train_y = []
        all_train_pred = []
        for batch_x_enc, batch_x_dec, batch_y in train_loader:
            batch_x_enc, batch_x_dec, batch_y = batch_x_enc.to(device), batch_x_dec.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x_enc, batch_x_dec)
            loss = criterion(predictions, batch_y)
            

            
            loss.backward()
            
            # Gradient Clipping pour stabiliser le LSTM
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            all_train_y.append(batch_y.detach().cpu().numpy())
            all_train_pred.append(predictions.detach().cpu().numpy())
            
        train_loss /= len(train_loader)
        train_r2 = r2_score(np.concatenate(all_train_y).flatten(), np.concatenate(all_train_pred).flatten())
        

        
        # Validation
        model.eval()
        val_loss = 0
        all_val_y = []
        all_val_pred = []
        with torch.no_grad():
            for batch_x_enc, batch_x_dec, batch_y in val_loader:
                batch_x_enc, batch_x_dec, batch_y = batch_x_enc.to(device), batch_x_dec.to(device), batch_y.to(device)
                predictions = model(batch_x_enc, batch_x_dec)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                all_val_y.append(batch_y.cpu().numpy())
                all_val_pred.append(predictions.cpu().numpy())
                
        
        val_loss /= len(val_loader)
        
        val_y_flat = np.concatenate(all_val_y).flatten()
        val_pred_flat = np.concatenate(all_val_pred).flatten()
        
        val_r2 = r2_score(val_y_flat, val_pred_flat)
        # Calcul de la VRAIE RMSE (et non la racine de la Huber Loss)
        val_true_mse = np.mean((val_y_flat - val_pred_flat)**2)
        val_rmse = math.sqrt(val_true_mse)
        
        # Update Scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Huber: {train_loss:.4f} (R²: {train_r2:.3f}) - Val Huber: {val_loss:.4f} (R²: {val_r2:.3f}) - Val RMSE: {val_rmse:.4f}")
        
        try:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'val_rmse': val_rmse
            }, step=epoch)
        except Exception:
            pass
            
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping déclenché après {epoch+1} epochs ! (Patience: {patience})")
                break
            
    print(f"Meilleure Val Loss: {best_val_loss:.4f}")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_val_loss, train_losses, val_losses, train_r2_scores, val_r2_scores

def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle LSTM")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                        help="Forcer l'exécution sur 'cpu' ou 'cuda'")
    args = parser.parse_args()

    print("\n=======================================================")
    print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT DU MODÈLE LSTM (SEQ2SEQ)")
    print("=======================================================")
    
    device = get_device(args.device)
    print(f"Device: {device}")
    
    X_enc_train, X_dec_train, Y_train = [], [], []
    X_enc_val, X_dec_val, Y_val = [], [], []
    X_enc_test, X_dec_test, Y_test = [], [], []
    
    # 1. Préparation des données (Découpage Temporel Strict)
    for site in SITES_PILOTES:
        print(f"\n--- Traitement des séquences pour {site} ---")
        
        # === TEST SET (Open-Meteo pour l'évaluation en conditions réelles, tous les sites) ===
        df_test_om = build_continuous_dataset(site, num_points=50, start_date="2024-01-01", end_date="2025-12-31", include_openmeteo=True)
        if not df_test_om.empty:
            xe, xd, y = create_sequences(df_test_om, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST, use_true_forecast=True, add_noise=False)
            X_enc_test.append(xe); X_dec_test.append(xd); Y_test.append(y)
            
        if site != "Gebesee":
            # === TRAIN SET (ERA5 Exact pour la théorie + ERA5 Bruité pour la robustesse) ===
            df_train_era = build_continuous_dataset(site, num_points=50, start_date="2021-01-01", end_date="2022-12-31", include_openmeteo=False)
            if not df_train_era.empty:
                # 1. Théorie (Exact)
                xe, xd, y = create_sequences(df_train_era, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST, use_true_forecast=False, add_noise=False)
                X_enc_train.append(xe); X_dec_train.append(xd); Y_train.append(y)
                # 2. Robustesse (Bruité)
                xe, xd, y = create_sequences(df_train_era, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST, use_true_forecast=False, add_noise=True)
                X_enc_train.append(xe); X_dec_train.append(xd); Y_train.append(y)
                
            # === VAL SET (ERA5 Bruité pour simuler la difficulté) ===
            df_val_era = build_continuous_dataset(site, num_points=50, start_date="2023-01-01", end_date="2023-12-31", include_openmeteo=False)
            if not df_val_era.empty:
                xe, xd, y = create_sequences(df_val_era, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST, use_true_forecast=False, add_noise=True)
                X_enc_val.append(xe); X_dec_val.append(xd); Y_val.append(y)
            
    if len(X_enc_train) == 0:
        print("Erreur : Aucun dataset d'entraînement n'a pu être construit.")
        return
        
    X_enc_train = np.concatenate(X_enc_train, axis=0)
    X_dec_train = np.concatenate(X_dec_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    
    X_enc_val = np.concatenate(X_enc_val, axis=0)
    X_dec_val = np.concatenate(X_dec_val, axis=0)
    Y_val = np.concatenate(Y_val, axis=0)
    
    X_enc_test = np.concatenate(X_enc_test, axis=0)
    X_dec_test = np.concatenate(X_dec_test, axis=0)
    Y_test = np.concatenate(Y_test, axis=0)
    
    # ---------------------------------------------------------
    # NORMALISATION (Z-SCORE SCALER) - Très important pour LSTM
    # ---------------------------------------------------------
    scaler_enc = StandardScaler()
    scaler_dec = StandardScaler()
    scaler_y = StandardScaler()
    
    # Reshape pour fiter les scalers (N*14, F)
    N_train, L_enc, F_enc = X_enc_train.shape
    _, L_dec, F_dec = X_dec_train.shape
    
    X_enc_train_flat = X_enc_train.reshape(-1, F_enc)
    X_dec_train_flat = X_dec_train.reshape(-1, F_dec)
    Y_train_flat = Y_train.reshape(-1, 1)
    
    # Fit & Transform
    X_enc_train = scaler_enc.fit_transform(X_enc_train_flat).reshape(N_train, L_enc, F_enc)
    X_dec_train = scaler_dec.fit_transform(X_dec_train_flat).reshape(N_train, L_dec, F_dec)
    Y_train = scaler_y.fit_transform(Y_train_flat).reshape(N_train, L_dec, 1)
    
    # Transform sur la Validation
    N_val = X_enc_val.shape[0]
    X_enc_val = scaler_enc.transform(X_enc_val.reshape(-1, F_enc)).reshape(N_val, L_enc, F_enc)
    X_dec_val = scaler_dec.transform(X_dec_val.reshape(-1, F_dec)).reshape(N_val, L_dec, F_dec)
    Y_val = scaler_y.transform(Y_val.reshape(-1, 1)).reshape(N_val, L_dec, 1)
    
    # Transform sur le Test
    N_test = X_enc_test.shape[0]
    X_enc_test = scaler_enc.transform(X_enc_test.reshape(-1, F_enc)).reshape(N_test, L_enc, F_enc)
    X_dec_test = scaler_dec.transform(X_dec_test.reshape(-1, F_dec)).reshape(N_test, L_dec, F_dec)
    Y_test = scaler_y.transform(Y_test.reshape(-1, 1)).reshape(N_test, L_dec, 1)
    
    # Sauvegarde des scalers pour l'inférence
    out_dir = os.path.join(OUTPUT_DIR, "Modeles_ML")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(scaler_enc, os.path.join(out_dir, "scaler_enc.pkl"))
    joblib.dump(scaler_dec, os.path.join(out_dir, "scaler_dec.pkl"))
    joblib.dump(scaler_y, os.path.join(out_dir, "scaler_y.pkl"))
    print("✅ Scalers sauvegardés.")
    
    print("\n--- Dimensions Globales du Dataset ---")
    print(f"Train - Enc: {X_enc_train.shape}, Dec: {X_dec_train.shape}, Y: {Y_train.shape}")
    print(f"Val   - Enc: {X_enc_val.shape}, Dec: {X_dec_val.shape}, Y: {Y_val.shape}")
    print(f"Test  - Enc: {X_enc_test.shape}, Dec: {X_dec_test.shape}, Y: {Y_test.shape}")
    
    # Création des DataLoaders
    # Calcul dynamique du batch size (viser ~300 itérations par époque, borné entre 64 et 8192)
    n_samples = len(X_enc_train)
    target_batch_size = max(64, min(n_samples // 300, 8192))
    # Arrondir à la puissance de 2 la plus proche (optimisation mémoire GPU)
    batch_size = int(2 ** np.round(np.log2(target_batch_size)))
    print(f"\n🔄 Batch Size calculé dynamiquement : {batch_size} (arrondi à la puissance de 2 pour {n_samples} séquences)")
    train_dataset = TensorDataset(torch.tensor(X_enc_train, dtype=torch.float32), torch.tensor(X_dec_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_enc_val, dtype=torch.float32), torch.tensor(X_dec_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_enc_test, dtype=torch.float32), torch.tensor(X_dec_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Instanciation du modèle
    input_dim_enc = X_enc_train.shape[2] # 6 features
    input_dim_dec = X_dec_train.shape[2] # 5 features
    output_dim = 1 # Prédiction de l'ET univariée
    
    model = Seq2SeqLSTM(
        encoder_input_dim=input_dim_enc,
        decoder_input_dim=input_dim_dec,
        hidden_dim=LSTM_HIDDEN_DIM,
        output_dim=output_dim,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT
    ).to(device)
    
    # --- MLFLOW INIT ---
    # MLflow stocke par défaut dans ./mlruns/ (relatif au CWD)
    mlflow.set_experiment("ET_LSTM_Forecasting")
    
    # On démarre la Run MLflow
    mlflow.start_run(run_name="L1_Dynamique_Test")
    mlflow.log_params({
        "lookback": LSTM_LOOKBACK,
        "forecast": LSTM_FORECAST,
        "batch_size": batch_size,
        "hidden_dim": LSTM_HIDDEN_DIM,
        "num_layers": LSTM_NUM_LAYERS,
        "dropout": LSTM_DROPOUT,
        "lr": LSTM_LR,
        "regularization": "L1 Dynamique"
    })
    
    # 3. Entraînement
    print("\n--- Début de l'entraînement ---")
    model, best_val_loss, train_losses, val_losses, train_r2_scores, val_r2_scores = train_model(model, train_loader, val_loader, num_epochs=LSTM_EPOCHS, lr=LSTM_LR, device=device)
    
    # Tracé de la learning curve
    out_dir_plot = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "LSTM_Evaluation")
    os.makedirs(out_dir_plot, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss (ShapeLoss)")
    plt.plot(val_losses, label="Validation Loss (ShapeLoss)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (ShapeLoss)")
    plt.title("Learning Curve du Modèle LSTM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir_plot, "1_Learning_Curve.png"), dpi=300)
    plt.close()
    
    # 4. Sauvegarde
    out_dir = os.path.join(OUTPUT_DIR, "Modeles_ML")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "lstm_et_forecaster.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Modèle sauvegardé avec succès dans : {model_path}")
    
    # 5. Évaluation finale sur le Test Set
    model.eval()
    test_loss = 0.0
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for x_enc_b, x_dec_b, y_b in test_loader:
            x_enc_b = x_enc_b.to(device)
            x_dec_b = x_dec_b.to(device)
            y_b = y_b.to(device)
            
            predictions = model(x_enc_b, x_dec_b)
            loss = criterion(predictions, y_b)
            test_loss += loss.item() * x_enc_b.size(0)
    
    test_loss /= len(test_loader.dataset)
    print(f"Performance Validation (MSE) sur 2024 : {best_val_loss:.4f}")
    print(f"Performance Test Finale (MSE) sur 2025 (données pures non vues) : {test_loss:.4f}")
    
    # 6. Génération automatique des graphiques
    try:
        from Traitement.LSTM_Forecasting.plot_predictions import plot_predictions
        print("\n==========================================")
        print("Lancement automatique des tracés (plot_predictions.py)...")
        print("==========================================")
        plot_predictions(model=model, device=device)
        
        # Enregistrer le dossier des graphiques dans MLflow
        mlflow.log_artifacts(out_dir_plot, artifact_path="Graphiques")
    except Exception as e:
        print(f"Erreur lors de la génération automatique des graphiques : {e}")
        
    mlflow.end_run()

if __name__ == "__main__":
    main()
