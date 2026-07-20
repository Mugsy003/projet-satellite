import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "Outputs")
CSV_ERA5 = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons", "Comparaison_ET_ICOS_vs_ERA5.csv")
CSV_PURE = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons", "Comparaison_ET_Landsat_vs_PureICOS.csv")
CSV_ERA5_B10 = os.path.join(OUTPUTS_DIR, "Resultats_ET_TTME_ERA5_B10.csv")
CSV_ERA5_DS = os.path.join(OUTPUTS_DIR, "Resultats_ET_TTME_ERA5_DS.csv")
OUT_DIR = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons")

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    if not os.path.exists(CSV_ERA5):
        print(f"❌ Le fichier {CSV_ERA5} n'existe pas.")
        return
        
    df_era5 = pd.read_csv(CSV_ERA5)
    df_era5['Date'] = pd.to_datetime(df_era5['Date'])
    
    # Charger les données pure ICOS si disponibles
    has_pure = False
    if os.path.exists(CSV_PURE):
        df_pure = pd.read_csv(CSV_PURE)
        df_pure['Date'] = pd.to_datetime(df_pure['Date'])
        # Fusionner sur Site et Date
        df = pd.merge(df_era5, df_pure[['Site', 'Date', 'ET_pure_ICOS (mm/h)']], on=['Site', 'Date'], how='left')
        has_pure = True
    else:
        df = df_era5
        
    has_b10 = False
    if os.path.exists(CSV_ERA5_B10):
        df_b10 = pd.read_csv(CSV_ERA5_B10)
        df_b10['Date'] = pd.to_datetime(df_b10['Date'])
        df_b10 = df_b10.rename(columns={'ET_pixel (mm/h)': 'ET_pixel (mm/h)_B10'})
        df = pd.merge(df, df_b10[['Site', 'Date', 'ET_pixel (mm/h)_B10']], on=['Site', 'Date'], how='left')
        has_b10 = True
        
    has_ds = False
    if 'ET_pixel (mm/h)_ERA5_DS' in df.columns:
        has_ds = True
        

    var_et_icos = 'ET_pixel (mm/h)_ICOS'
    var_et_era5 = 'ET_pixel (mm/h)_ERA5'
    var_et_pure = 'ET_pure_ICOS (mm/h)' if has_pure else None
    var_et_b10 = 'ET_pixel (mm/h)_B10' if has_b10 else None
    var_et_ds = 'ET_pixel (mm/h)_ERA5_DS' if has_ds else None
    
    sites = df['Site'].unique()
    print(f"📊 Génération des graphiques (3 courbes) pour {len(sites)} sites...")
    
    for site in sites:
        df_site = df[df['Site'] == site].copy()
        df_site = df_site.sort_values('Date')
        
        if len(df_site) < 2:
            print(f"⚠️ Pas assez de données pour le site {site}. Ignoré.")
            continue
            
        et_icos = df_site[var_et_icos]
        et_era5 = df_site[var_et_era5]
        et_pure = df_site[var_et_pure] if has_pure else pd.Series([np.nan]*len(df_site))
        et_b10 = df_site[var_et_b10] if has_b10 else pd.Series([np.nan]*len(df_site))
        et_ds = df_site[var_et_ds] if has_ds else pd.Series([np.nan]*len(df_site))
        
        # Définition de la référence (Pure ICOS si dispo, sinon ICOS Landsat)
        ref_name = "Pure ICOS (in-situ)" if has_pure else "ICOS (LST Landsat)"
        et_ref = et_pure if has_pure else et_icos
        
        # Fonction pour calculer les métriques
        def calc_metrics(et_model):
            mask = et_ref.notna() & et_model.notna()
            if mask.sum() >= 2:
                r2 = r2_score(et_ref[mask], et_model[mask])
                rmse = np.sqrt(mean_squared_error(et_ref[mask], et_model[mask]))
                bias = np.mean(et_model[mask] - et_ref[mask])
                return f"{r2:.2f}", f"{rmse:.3f}", f"{bias:.3f}"
            return "N/A", "N/A", "N/A"
            
        r2_era5, rmse_era5, bias_era5 = calc_metrics(et_era5)
        
        # Préparation du tableau
        cell_text = [
            ["ERA5 brute", r2_era5, rmse_era5, bias_era5]
        ]
        
        if has_ds:
            r2_ds, rmse_ds, bias_ds = calc_metrics(et_ds)
            cell_text.append(["ERA5 DS", r2_ds, rmse_ds, bias_ds])
            
        if has_b10:
            r2_b10, rmse_b10, bias_b10 = calc_metrics(et_b10)
            cell_text.append(["ERA5 (LST B10)", r2_b10, rmse_b10, bias_b10])
            
        # 1. Tableau des performances (à la place du scatter plot)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax1 = axes[0]
        ax1.axis('tight')
        ax1.axis('off')
        
        col_labels = ["Modèle", "R²", "RMSE (mm/h)", "Biais (mm/h)"]
        table = ax1.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
        table.scale(1, 2)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        
        # Mettre les en-têtes en gras et colorer
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')
                
        ax1.set_title(f"Performances vs {ref_name}", fontsize=13, weight='bold', pad=20)
        
        # 2. Série temporelle (3 courbes)
        ax2 = axes[1]
        
        # Courbe 1 : Le "Ground Truth" théorique (Pure ICOS)
        if has_pure:
            mask_pure = et_pure.notna()
            # On relie les points valides
            ax2.plot(df_site['Date'][mask_pure], et_pure[mask_pure], marker='D', linestyle='-', color='purple', label='ET Pure ICOS (LST in-situ)', linewidth=2.5)
1            
        # Courbe 2 : ICOS + Landsat
        ax2.plot(df_site['Date'], et_icos, marker='o', linestyle='-', color='forestgreen', label='ET ICOS (LST Landsat)', linewidth=2, alpha=0.8)
        
        # Courbe 3 : ERA5 + Landsat
        ax2.plot(df_site['Date'], et_era5, marker='s', linestyle='--', color='darkorange', label='ET ERA5 (LST Landsat DMS)', linewidth=2, alpha=0.8)
        
        # Courbe ERA5 DS
        if has_ds:
            ax2.plot(df_site['Date'], et_ds, marker='v', linestyle='-', color='dodgerblue', label='ET ERA5 DS (LST Landsat DMS)', linewidth=2, alpha=0.9)
        
        # Courbe 4 : ERA5 + Landsat B10
        if has_b10:
            ax2.plot(df_site['Date'], et_b10, marker='^', linestyle='-.', color='red', label='ET ERA5 (LST brute B10)', linewidth=2, alpha=0.8)
            
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylabel("Évapotranspiration (mm/h)", fontsize=11)
        ax2.set_title("Évolution Temporelle", fontsize=13)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize=11)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.suptitle(f"Comparaison de l'Évapotranspiration - Site : {site}", fontsize=16, y=1.05)
        plt.tight_layout()
        
        out_png = os.path.join(OUT_DIR, f"Comparaison_ET_{site}.png")
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
