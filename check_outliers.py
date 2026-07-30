import pandas as pd
import numpy as np

def check():
    path = "Outputs_ICOS/donnees_icos_LE_Gebesee.csv"
    df = pd.read_csv(path, index_col='TIMESTAMP', parse_dates=True)
    
    # Filtrer pour 2024
    df_2024 = df.loc["2024-01-01":"2024-12-31"]
    
    # Récupérer les mesures autour de 10:30
    # On resample pour garder uniquement les valeurs entre 10:00 et 11:00 par exemple
    df_1030 = df_2024.between_time('10:00', '11:00')
    
    # Agréger par jour
    daily_le = df_1030['LE_Consolide'].resample('D').mean()
    
    # Convertir en ET mm/jour
    daily_et = daily_le * 3600.0 * 24.0 / 2.45e6
    
    daily_et = daily_et.dropna()
    
    print(f"Nombre de jours valides en 2024: {len(daily_et)}")
    print(f"ET Minimum ICOS (mm/j): {daily_et.min():.2f}")
    print(f"ET Maximum ICOS (mm/j): {daily_et.max():.2f}")
    
    # Afficher les 5 valeurs les plus hautes
    print("\nTop 5 valeurs max:")
    print(daily_et.nlargest(5))
    
    # Afficher les 5 valeurs les plus basses
    print("\nTop 5 valeurs min:")
    print(daily_et.nsmallest(5))

if __name__ == "__main__":
    check()
