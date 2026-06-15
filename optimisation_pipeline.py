import optuna
import json
import subprocess
import os
import pandas as pd
import numpy as np

CSV_PATH = r"Outputs\Validation_Saisonniere_LST.csv"

# --- Paramètres de période d'évaluation ---
# Format: 'YYYY-MM-DD'. Mettre None pour ne pas filtrer.
START_DATE = "2022-12-31" 
END_DATE = "2024-01-01"
# ------------------------------------------

def objective(trial):
    # 1. Suggerer les hyperparamètres
    model_type = trial.suggest_categorical("model_type", ["RandomForest", "LightGBM"])
    
    hyperparams = {"model_type": model_type}
    
    if model_type == "RandomForest":
        hyperparams["n_estimators"] = trial.suggest_int("n_estimators", 50, 1000, step=25)
        hyperparams["max_depth"] = trial.suggest_int("max_depth", 5, 100)
        hyperparams["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 50)
        hyperparams["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 10)
        hyperparams["max_features"] = trial.suggest_float("max_features", 0.1, 1.0, step=0.1)
    else:
        hyperparams["n_estimators"] = trial.suggest_int("n_estimators", 50, 1000, step=25)
        hyperparams["max_depth"] = trial.suggest_int("max_depth", 5, 100)
        hyperparams["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
        hyperparams["num_leaves"] = trial.suggest_int("num_leaves", 2, 512)
        hyperparams["subsample"] = trial.suggest_float("subsample", 0.1, 1.0, step=0.1)
        
    # Ecrire dans un fichier temporaire
    with open("hyperparams_tmp.json", "w") as f:
        json.dump(hyperparams, f)
        
    print(f"\n{'='*50}")
    print(f"--- TRIAL {trial.number} ---")
    print(f"Modèle : {model_type}")
    print(f"Paramètres : {hyperparams}")
    print(f"{'='*50}\n")
    
    try:
        # Lancer le Sharpening (DMS)
        subprocess.run(["venv\\Scripts\\python.exe", "-m", "Transform.Landsat.dms_sharpening_landsat"], check=True)
        
        # Lancer la Comparaison
        subprocess.run(["venv\\Scripts\\python.exe", "comparaison_ICOS.py"], check=True)
        
        # Evaluer la RMSE
        if not os.path.exists(CSV_PATH):
            return float('inf')
            
        df = pd.read_csv(CSV_PATH)
        
        # Filtrer sur la période souhaitée
        if START_DATE is not None or END_DATE is not None:
            # On s'assure que la colonne est au format datetime
            df['Date_Satellite'] = pd.to_datetime(df['Date_Satellite'])
            if START_DATE is not None:
                df = df[df['Date_Satellite'] >= pd.to_datetime(START_DATE)]
            if END_DATE is not None:
                df = df[df['Date_Satellite'] <= pd.to_datetime(END_DATE)]
                
        # On calcule la RMSE pour le DMS (Landsat)
        mask = pd.notna(df['LST_Sat_DMS (°C)']) & pd.notna(df['ICOS_LST (°C)'])
        y_pred = df.loc[mask, 'LST_Sat_DMS (°C)']
        y_true = df.loc[mask, 'ICOS_LST (°C)']
        
        if len(y_pred) == 0:
            return float('inf')
            
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        print(f"\n--> RMSE OBTENUE POUR LE TRIAL {trial.number} : {rmse:.3f}°C\n")
        
        return rmse
        
    except Exception as e:
        print(f"Erreur durant le Trial {trial.number}: {e}")
        return float('inf')

if __name__ == "__main__":
    print("Démarrage de l'optimisation globale Optuna...")
    # Sauvegarde dans une base de données locale pour permettre la reprise en cas de coupure
    study = optuna.create_study(
        study_name="optimisation_satellite",
        storage="sqlite:///optuna_study.db", 
        load_if_exists=True, 
        direction="minimize"
    )
    study.optimize(objective, n_trials=50)
    
    print("\n" + "="*50)
    print("OPTIMISATION TERMINÉE !")
    print("="*50)
    print("Meilleurs Paramètres trouvés :")
    print(json.dumps(study.best_params, indent=4))
    print(f"Meilleure RMSE : {study.best_value:.3f}°C")
    
    # Sauvegarder les meilleurs paramètres
    with open("best_hyperparams.json", "w") as f:
        json.dump(study.best_params, f)
        
    # Nettoyage
    if os.path.exists("hyperparams_tmp.json"):
        os.remove("hyperparams_tmp.json")
