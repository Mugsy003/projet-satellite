"""
calcul_cwsi.py
Script pour calculer le CWSI (Crop Water Stress Index) à partir des résultats LST 
et des données météorologiques ICOS (Température de l'air et VPD).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
# pyrefly: ignore [missing-import]
from icoscp.dobj import Dobj

from config import PIDS_ICOS, OUTPUT_DIR

# --- PARAMÈTRES CWSI ---
# Upper Baseline spécifique à chaque site (Maximum-stressed baseline en °C)
# Vous pouvez ajuster ces valeurs selon la culture.
UPPER_BASELINE_SITE = {
    "Gebesee": 4.0,     # Exemple : +4.0°C pour des grandes cultures agricoles
    "Selhausen": 4.0,   # Exemple : +4.0°C
    "Lamasquere": 3.0,
    "Lonzee": 3.0,
    "Greece": 5.0,
    "Italy": 5.0
}
DEFAULT_UPPER = 4.0

def load_icos_meteo(site, pid):
    """
    Télécharge les données ICOS et extrait la Température de l'air (Ta) et le VPD.
    """
    try:
        dobj = Dobj(pid)
        if not dobj.valid:
            return None
        
        df = dobj.data
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df.replace([-9.99, -999.0, -9999.0], np.nan, inplace=True)
        
        # Chercher la colonne de Température de l'air (Ta)
        ta_cols = [c for c in dobj.colNames if c.startswith('TA_')]
        ta_col = 'TA_F' if 'TA_F' in ta_cols else (ta_cols[0] if ta_cols else None)
        
        # Chercher la colonne VPD
        vpd_cols = [c for c in dobj.colNames if c.startswith('VPD_')]
        vpd_col = 'VPD_F' if 'VPD_F' in vpd_cols else (vpd_cols[0] if vpd_cols else None)
        
        if not ta_col or not vpd_col:
            print(f"Colonnes TA ou VPD manquantes dans ICOS pour {site}.")
            return None
            
        return df[['TIMESTAMP', ta_col, vpd_col]].rename(columns={
            ta_col: 'Ta_ICOS',
            vpd_col: 'VPD_ICOS'
        })
        
    except Exception as e:
        print(f"Erreur ICOS pour {site} : {e}")
        return None

def main():
    csv_path = os.path.join(OUTPUT_DIR, "Validation_Saisonniere_LST.csv")
    if not os.path.exists(csv_path):
        print(f"Le fichier {csv_path} est introuvable.")
        return
        
    print(f"Chargement de {csv_path}...")
    df_results = pd.read_csv(csv_path)
    df_results.replace("N/A", np.nan, inplace=True)
    
    # On utilisera le DMS comme LST de référence pour le CWSI (le plus constant)
    lst_col = 'LST_Sat_DMS (°C)'
    if lst_col not in df_results.columns:
        print(f"La colonne {lst_col} est manquante.")
        return
        
    df_results[lst_col] = pd.to_numeric(df_results[lst_col], errors='coerce')
    df_results = df_results.dropna(subset=[lst_col, 'Heure_ICOS_Retenue'])
    
    # Préparer les nouvelles colonnes
    df_results['Ta_ICOS (°C)'] = np.nan
    df_results['VPD_ICOS (hPa)'] = np.nan
    
    # 1. Récupérer les données ICOS pour chaque site
    sites_presents = df_results['Site'].unique()
    for site in sites_presents:
        if site not in PIDS_ICOS:
            print(f"Pas de PID ICOS pour le site {site}. On ignore.")
            continue
            
        print(f"Recuperation des donnees ICOS pour {site}...")
        df_meteo = load_icos_meteo(site, PIDS_ICOS[site])
        
        if df_meteo is not None:
            # Mettre à jour les lignes du CSV
            masque_site = df_results['Site'] == site
            
            for idx, row in df_results[masque_site].iterrows():
                heure_str = row['Date_Satellite'].split(' ')[0] + ' ' + row['Heure_ICOS_Retenue']
                try:
                    dt_cible = pd.to_datetime(heure_str)
                    
                    # Trouver la ligne correspondante
                    diffs = abs(df_meteo['TIMESTAMP'] - dt_cible)
                    idx_min = diffs.idxmin()
                    
                    if diffs[idx_min] <= pd.Timedelta(minutes=30):
                        ta = df_meteo.loc[idx_min, 'Ta_ICOS']
                        vpd = df_meteo.loc[idx_min, 'VPD_ICOS']
                        
                        df_results.at[idx, 'Ta_ICOS (°C)'] = ta
                        df_results.at[idx, 'VPD_ICOS (hPa)'] = vpd
                        
                except Exception as e:
                    print(f"Erreur d'association pour la ligne {idx}: {e}")
                    continue

    # Filtrer où on a toutes les variables
    df_cwsi = df_results.dropna(subset=['Ta_ICOS (°C)', 'VPD_ICOS (hPa)', lst_col]).copy()
    
    if df_cwsi.empty:
        print("Aucune ligne n'a les donnees completes (LST, Ta, VPD) pour calculer le CWSI.")
        return
        
    df_cwsi['Tc-Ta'] = df_cwsi[lst_col] - df_cwsi['Ta_ICOS (°C)']
    df_cwsi['CWSI'] = np.nan
    
    # 2. Calcul du CWSI par site
    print("\nCalcul des Baselines et du CWSI par site :")
    
    # On prépare un graphique pour montrer les régressions
    num_sites = len(df_cwsi['Site'].unique())
    fig, axes = plt.subplots(1, num_sites, figsize=(6 * num_sites, 5), squeeze=False)
    axes = axes.flatten()
    
    for i, site in enumerate(df_cwsi['Site'].unique()):
        masque = df_cwsi['Site'] == site
        df_site = df_cwsi[masque].copy()
        
        if len(df_site) < 3:
            print(f"Pas assez de points pour calculer la baseline empirique pour {site} (N={len(df_site)}).")
            continue
            
        vpd_vals = df_site['VPD_ICOS (hPa)'].values
        diff_vals = df_site['Tc-Ta'].values
        
        # Calcul empirique de la Lower Baseline (a + b * VPD)
        # On utilise une régression linéaire simple sur les données disponibles
        # Note: Idéalement, cela devrait être fait sur les points les plus bas (plantes non stressées)
        slope, intercept, r_value, p_value, std_err = linregress(vpd_vals, diff_vals)
        
        # Upper baseline
        upper_limit = UPPER_BASELINE_SITE.get(site, DEFAULT_UPPER)
        
        print(f"  [{site}] Lower Baseline : Tc-Ta = {slope:.3f} * VPD + {intercept:.3f} (R2={r_value**2:.2f})")
        print(f"  [{site}] Upper Baseline : {upper_limit}°C")
        
        # Calcul du CWSI
        for idx, row in df_site.iterrows():
            vpd = row['VPD_ICOS (hPa)']
            tc_ta = row['Tc-Ta']
            
            lower_val = intercept + slope * vpd
            upper_val = upper_limit
            
            # Formule du CWSI
            cwsi_val = (tc_ta - lower_val) / (upper_val - lower_val)
            
            # Limiter mathématiquement entre 0 et 1
            cwsi_val = max(0.0, min(1.0, cwsi_val))
            df_cwsi.at[idx, 'CWSI'] = round(cwsi_val, 3)
            
        # Tracé du graphique
        ax = axes[i]
        ax.scatter(vpd_vals, diff_vals, color='blue', label='Données (Tc-Ta)')
        
        # Tracé des baselines
        vpd_range = np.linspace(min(vpd_vals)*0.9, max(vpd_vals)*1.1, 100)
        ax.plot(vpd_range, intercept + slope * vpd_range, 'g--', label='Lower Baseline')
        ax.axhline(y=upper_limit, color='r', linestyle='--', label='Upper Baseline')
        
        ax.set_title(f"Baselines CWSI - {site}")
        ax.set_xlabel("VPD (hPa)")
        ax.set_ylabel("Tc - Ta (°C)")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    # Sauvegarde des graphiques
    plot_path = os.path.join(OUTPUT_DIR, "CWSI_Baselines.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    # 3. Export du CSV
    out_csv = os.path.join(OUTPUT_DIR, "Validation_CWSI.csv")
    df_cwsi.to_csv(out_csv, index=False)
    
    print(f"\nTermine ! Resultats sauvegardes dans : {out_csv}")
    print(f"Graphiques sauvegardes dans : {plot_path}")
    
    # Petit aperçu
    print("\nAperçu des résultats :")
    print(df_cwsi[['Site', 'Date_Satellite', 'Ta_ICOS (°C)', 'VPD_ICOS (hPa)', 'Tc-Ta', 'CWSI']].head(10))

if __name__ == "__main__":
    main()
