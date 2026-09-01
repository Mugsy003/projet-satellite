import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import OUTPUT_DIR, SITES_PILOTES
from Traitement.LSTM_Forecasting.dataset_prep import build_continuous_dataset

def main():
    print("Analyse de l'erreur des prévisions Open-Meteo par Horizon (J+1 à J+7)")
    
    # On va stocker les erreurs (Prevision - Realite) pour Ta et RH
    errors_ta = []
    errors_rh = []
    horizons = []
    
    for site in SITES_PILOTES:
        print(f"--- Traitement de {site} ---")
        df = build_continuous_dataset(site, num_points=10, start_date="2024-01-01", end_date="2025-12-31", include_openmeteo=True)
        if df.empty:
            continue
            
        print(f"Analyse pour le site : {site}")
        
        # Le groupement par Point_ID est nécessaire
        for pt_id, df_pt in df.groupby('Point_ID'):
            df_pt = df_pt.sort_values('Date').reset_index(drop=True)
            
            # Pour chaque ligne T, on a les prévisions J1 à J7 faites ce jour-là
            # On veut comparer Ta_fcst_Jd avec le vrai Ta au jour T+d
            for i in range(len(df_pt) - 7):
                row_T = df_pt.iloc[i]
                
                for d in range(1, 8):
                    target_row = df_pt.iloc[i + d]
                    
                    # Prevision
                    ta_fcst = row_T[f'Ta_fcst_J{d}']
                    rh_fcst = row_T[f'RH_fcst_J{d}']
                    
                    # Realite
                    ta_real = target_row['Ta']
                    rh_real = target_row['RH']
                    
                    # Erreur = Prevu - Reel
                    err_ta = ta_fcst - ta_real
                    err_rh = rh_fcst - rh_real
                    
                    errors_ta.append(err_ta)
                    errors_rh.append(err_rh)
                    horizons.append(f'J+{d}')
                    
    if len(errors_ta) == 0:
        print("Aucune donnée disponible pour l'analyse.")
        return
        
    df_errors = pd.DataFrame({
        'Horizon': horizons,
        'Erreur_Ta': errors_ta,
        'Erreur_RH': errors_rh
    })
    
    # Création des graphiques (Boxplots)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Boxplot Erreur Température
    sns.boxplot(data=df_errors, x='Horizon', y='Erreur_Ta', ax=axes[0], palette="coolwarm", showfliers=False)
    axes[0].set_title("Erreur de Prévision Température (Prévu - Réel)", fontsize=14)
    axes[0].set_ylabel("Erreur (°C)")
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1.5)
    
    # Boxplot Erreur Humidité
    sns.boxplot(data=df_errors, x='Horizon', y='Erreur_RH', ax=axes[1], palette="YlGnBu", showfliers=False)
    axes[1].set_title("Erreur de Prévision Humidité Relative (Prévu - Réel)", fontsize=14)
    axes[1].set_ylabel("Erreur (%)")
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5)
    
    plt.tight_layout()
    out_dir = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "LSTM_Evaluation")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "Erreurs_Previsions_Meteo.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    print(f"Graphique sauvegardé dans : {out_file}")
    
    # Affichage des statistiques (RMSE et Biais)
    print("\n--- Statistiques Globales par Horizon ---")
    stats = df_errors.groupby('Horizon').agg(
        Biais_Ta=('Erreur_Ta', 'mean'),
        RMSE_Ta=('Erreur_Ta', lambda x: np.sqrt(np.mean(x**2))),
        Biais_RH=('Erreur_RH', 'mean'),
        RMSE_RH=('Erreur_RH', lambda x: np.sqrt(np.mean(x**2)))
    )
    print(stats)

if __name__ == "__main__":
    main()
