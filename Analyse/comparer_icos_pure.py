import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Configuration des chemins
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Outputs")
COMPARE_DIR = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons")
os.makedirs(COMPARE_DIR, exist_ok=True)

FILE_ICOS = os.path.join(OUTPUTS_DIR, "Resultats_ET_TTME_ICOS.csv")
ICOS_METEO_DIR = os.path.join(os.path.dirname(OUTPUTS_DIR), "Outputs_ICOS")

LAMBDA_V = 2.45e6  # Chaleur latente de vaporisation (J/kg)

def main():
    print("=" * 60)
    print("   COMPARAISON ET : Landsat LST vs ICOS LST (Pure ICOS)")
    print("=" * 60)
    
    if not os.path.exists(FILE_ICOS):
        print(f"❌ Fichier introuvable : {FILE_ICOS}")
        return
        
    df = pd.read_csv(FILE_ICOS)
    print(f"✅ Fichier de base chargé : {len(df)} lignes.")
    
    # On va calculer la LST ICOS et l'ET ICOS Pure
    lst_icos_list = []
    et_pure_list = []
    le_pure_list = []
    
    for idx, row in df.iterrows():
        site = row['Site']
        date_str = row['Date']
        
        # 1. Trouver LST_ground dans les fichiers météo ICOS
        target_dt = pd.to_datetime(f"{date_str} 10:30:00")
        meteo_path = os.path.join(ICOS_METEO_DIR, f"donnees_icos_{site}.csv")
        
        lst_ground = np.nan
        if os.path.exists(meteo_path):
            df_meteo = pd.read_csv(meteo_path)
            df_meteo['TIMESTAMP'] = pd.to_datetime(df_meteo['TIMESTAMP'])
            diff = abs(df_meteo['TIMESTAMP'] - target_dt)
            mask = diff <= pd.Timedelta(minutes=60)
            if not df_meteo[mask].empty:
                idx_best = diff[mask].idxmin()
                lst_ground = df_meteo.loc[idx_best, 'LST_Calculee']
                
        lst_icos_list.append(lst_ground)
        
        # 2. Recalculer l'ET avec le modèle TTME en utilisant LST_ground à la place de lst_pixel
        if pd.isna(lst_ground) or pd.isna(row['fc_pixel']) or pd.isna(row['T_s_max (°C)']):
            et_pure_list.append(np.nan)
            le_pure_list.append(np.nan)
            continue
            
        fc = row['fc_pixel']
        Ta = row['Ta (°C)']
        Rn = row['Rn (W/m²)']
        Ts_max = row['T_s_max (°C)']
        Tc_max = row['T_c_max (°C)']
        
        # Paramètres géométriques du trapèze
        beta_w = Tc_max - Ts_max
        a = lst_ground - Ta
        T_warm_at_fc = Ts_max + beta_w * fc
        a_plus_b = T_warm_at_fc - Ta
        if a_plus_b > 0.1:
            ratio = np.clip(a / a_plus_b, 0.0, 1.0)
        else:
            ratio = np.nan
        beta_i = ratio * beta_w
        
        # Décomposition de LST
        Ts = lst_ground - beta_i * fc
        Tc = Ts + beta_i
        
        # Contraintes physiques
        Ts = max(Ts, Ta - 5)
        Tc = max(Tc, Ta - 5)
        
        # Flux de chaleur dans le sol
        cg_s = 0.315 if fc < 0.5 else 0.35
        cg_c = 0.05
        
        # L'énergie disponible pour chaque pôle pur
        G_pur_sol = cg_s * Rn
        G_pur_canopee = cg_c * Rn
        
        Rn_s_dispo = Rn - G_pur_sol
        Rn_c_dispo = Rn - G_pur_canopee
        
        # Calcul de la Chaleur Latente par interpolation linéaire
        if Ts_max > Ta:
            LE_s = Rn_s_dispo * (Ts_max - Ts) / (Ts_max - Ta)
        else:
            LE_s = 0.0
            
        if Tc_max > Ta:
            LE_c = Rn_c_dispo * (Tc_max - Tc) / (Tc_max - Ta)
        else:
            LE_c = 0.0
            
        # LE Total : mosaïque
        LE = fc * LE_c + (1 - fc) * LE_s
        energie_dispo = Rn - (fc * G_pur_canopee + (1 - fc) * G_pur_sol)
        
        LE = np.clip(LE, 0.0, energie_dispo)
        
        # ET en mm/h
        ET = LE * 3600.0 / LAMBDA_V
        
        et_pure_list.append(ET)
        le_pure_list.append(LE)
        
    df['LST_ICOS (°C)'] = lst_icos_list
    df['LE_pure_ICOS (W/m²)'] = le_pure_list
    df['ET_pure_ICOS (mm/h)'] = et_pure_list
    
    # Nettoyage pour la comparaison
    df_valid = df.dropna(subset=['ET_pixel (mm/h)', 'ET_pure_ICOS (mm/h)'])
    print(f"🔄 Comparaison possible sur {len(df_valid)} points.")
    
    if len(df_valid) == 0:
        print("⚠️ Aucune donnée valide trouvée.")
        return
        
    # Statistiques ET
    et_landsat = df_valid['ET_pixel (mm/h)']
    et_pure = df_valid['ET_pure_ICOS (mm/h)']
    
    lst_landsat = df_valid['LST_pixel (°C)']
    lst_pure = df_valid['LST_ICOS (°C)']
    
    bias_lst = np.mean(lst_pure - lst_landsat)
    
    rmse = np.sqrt(mean_squared_error(et_landsat, et_pure))
    bias = np.mean(et_pure - et_landsat)
    r2 = r2_score(et_landsat, et_pure)
    
    print("-" * 40)
    print("MÉTRIQUES (Pure ICOS vs Landsat-ICOS)")
    print("-" * 40)
    print(f"Biais LST = {bias_lst:.2f} °C")
    print(f"RMSE = {rmse:.4f} mm/h")
    print(f"Biais= {bias:.4f} mm/h")
    print(f"R²   = {r2:.4f}")
    
    # Sauvegarde CSV
    out_csv = os.path.join(COMPARE_DIR, "Comparaison_ET_Landsat_vs_PureICOS.csv")
    df_valid.to_csv(out_csv, index=False)
    
    # Graphique
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(et_landsat, et_pure, color='purple', alpha=0.7)
    
    min_val = min(et_landsat.min(), et_pure.min())
    max_val = max(et_landsat.max(), et_pure.max())
    margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'r--', label='1:1')
    
    ax.set_title("Comparaison ET : LST Landsat vs LST ICOS In-Situ", fontsize=14)
    ax.set_xlabel("ET avec LST Landsat (mm/h)", fontsize=12)
    ax.set_ylabel("ET avec LST In-Situ (Pure ICOS) (mm/h)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    textstr = f"N = {len(df_valid)}\nR² = {r2:.2f}\nRMSE = {rmse:.3f}\nBiais = {bias:.3f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)
            
    out_png = os.path.join(COMPARE_DIR, "Scatter_ET_Landsat_vs_PureICOS.png")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 Graphique sauvegardé : {out_png}")

if __name__ == "__main__":
    import sys
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
