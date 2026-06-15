# pyrefly: ignore [missing-import]
import optuna

try:
    study = optuna.load_study(study_name='optimisation_satellite', storage='sqlite:///optuna_study.db')
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    print(f"Total des trials lancés : {len(trials)}")
    print(f"Trials complétés (sans erreur) : {len(completed)}")
    
    if completed:
        print(f"Meilleur RMSE (best_value) : {study.best_value:.4f}")
        print("Meilleurs hyperparamètres :")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
    else:
        print("Aucun trial complété avec succès pour l'instant.")
except Exception as e:
    print(f"Erreur lors de la lecture de l'étude Optuna : {e}")
