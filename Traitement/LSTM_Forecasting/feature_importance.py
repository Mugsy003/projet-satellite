import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

# Ajuster le chemin pour importer depuis la racine
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import OUTPUT_DIR, LSTM_LOOKBACK, LSTM_FORECAST, LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT
from Traitement.LSTM_Forecasting.model_lstm import Seq2SeqLSTM
from Traitement.LSTM_Forecasting.train_lstm import get_device
from Traitement.LSTM_Forecasting.dataset_prep import build_continuous_dataset, create_sequences

class LSTMFeatureWrapper(nn.Module):
    """
    Wrapper pour que le modèle retourne un scalaire par exemple (la somme de l'ET prédite)
    ce qui est nécessaire pour l'algorithme Integrated Gradients.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x_enc, x_dec):
        out = self.model(x_enc, x_dec)
        # On somme sur l'horizon de prédiction pour avoir l'attribution globale
        return out.sum(dim=1).squeeze(-1)

def compute_feature_importance():
    print("=======================================================")
    print("📊 ANALYSE D'IMPORTANCE DES FEATURES (CAPTUM - IG)")
    print("=======================================================")
    
    device = get_device("auto")
    model_path = os.path.join(OUTPUT_DIR, "Modeles_ML", "lstm_et_forecaster.pt")
    out_dir = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "LSTM_Evaluation")
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable : {model_path}")
        return
        
    # Noms des features (Doit correspondre au modèle sauvegardé qui prenait 7 et 3 features)
    enc_features = ['NDVI', 'SAVI', 'NDWI', 'PT_SINRH_ET', 'Ta', 'RH', 'Rn']
    dec_features = ['Ta_fcst', 'RH_fcst', 'Rs_fcst']
    
    # 1. Charger l'échantillon de données (Année 2024 de Gebesee par exemple)
    print("Chargement des données de test (Gebesee 2024)...")
    df_val = build_continuous_dataset("Gebesee", num_points=15, start_date="2024-01-01", end_date="2024-12-31", include_openmeteo=True)
    if df_val.empty:
        print("❌ Aucune donnée trouvée pour 2024.")
        return
        
    xe, xd, _ = create_sequences(df_val, lookback=LSTM_LOOKBACK, forecast=LSTM_FORECAST, add_noise=True)
    
    # Prendre un échantillon pour ne pas exploser la RAM (ex: 500 séquences)
    np.random.seed(42)
    sample_indices = np.random.choice(len(xe), min(500, len(xe)), replace=False)
    xe_sample = xe[sample_indices, :, :7]  # On garde les 7 features
    xd_sample = xd[sample_indices, :, :3]  # Idem, 3 pour le decodeur
    
    # 2. Normalisation (Z-Score)
    # Pour Captum, l'idéal est de normaliser avec les mêmes stats que le modèle
    # Ici, pour simplifier, on va refaire un Z-Score local sur l'échantillon
    xe_mean = xe_sample.mean(axis=(0, 1), keepdims=True)
    xe_std = xe_sample.std(axis=(0, 1), keepdims=True) + 1e-8
    xe_scaled = (xe_sample - xe_mean) / xe_std
    
    xd_mean = xd_sample.mean(axis=(0, 1), keepdims=True)
    xd_std = xd_sample.std(axis=(0, 1), keepdims=True) + 1e-8
    xd_scaled = (xd_sample - xd_mean) / xd_std
    
    t_xe = torch.tensor(xe_scaled, dtype=torch.float32).to(device)
    t_xd = torch.tensor(xd_scaled, dtype=torch.float32).to(device)
    
    # 3. Charger le modèle
    print("Chargement du modèle LSTM...")
    model = Seq2SeqLSTM(
        encoder_input_dim=len(enc_features),
        decoder_input_dim=len(dec_features),
        hidden_dim=LSTM_HIDDEN_DIM,
        output_dim=1,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    wrapper = LSTMFeatureWrapper(model)
    ig = IntegratedGradients(wrapper)
    
    print("Lancement de Captum (Integrated Gradients)...")
    # baselines par défaut = 0 (ce qui correspond à la moyenne grâce au Z-Score)
    attributions = ig.attribute(inputs=(t_xe, t_xd))
    
    attr_enc, attr_dec = attributions
    
    # 4. Agréger les attributions
    # Prendre la moyenne absolue sur le batch et sur le temps
    mean_attr_enc = attr_enc.abs().mean(dim=(0, 1)).cpu().numpy()
    mean_attr_dec = attr_dec.abs().mean(dim=(0, 1)).cpu().numpy()
    
    # Combinaison : On va regrouper par nom de feature
    importance_dict = {}
    
    for i, f in enumerate(enc_features):
        importance_dict[f] = importance_dict.get(f, 0) + mean_attr_enc[i]
        
    for i, f in enumerate(dec_features):
        importance_dict[f] = importance_dict.get(f, 0) + mean_attr_dec[i]
        
    # Trier par importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features = [x[0] for x in sorted_features]
    scores = [x[1] for x in sorted_features]
    
    # 5. Graphique
    plt.figure(figsize=(10, 6))
    plt.bar(features, scores, color='teal')
    plt.ylabel('Attribution Absolue Moyenne (Captum IG)')
    plt.title('Importance des Variables Physiques (LSTM)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "feature_importance_captum.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"✅ Analyse terminée. Graphique généré : {out_path}")
    
    # Sauvegarde des scores en CSV
    df_scores = pd.DataFrame({'Feature': features, 'Importance': scores})
    df_scores.to_csv(os.path.join(out_dir, "feature_importance_scores.csv"), index=False)

if __name__ == "__main__":
    compute_feature_importance()
