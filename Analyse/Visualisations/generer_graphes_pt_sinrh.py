"""
Génération des graphiques de comparaison PT-SINRH par site.

Pour chaque site, génère un graphique avec :
  - Un tableau de performances (r², RMSE, Biais) pour TTME et PT-SINRH.
  - Une série temporelle comparant les courbes TTME Pure ICOS et PT-SINRH.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates

# ===========================================================================
# CHEMINS
# ===========================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "Outputs")
CSV_PT = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons", "Comparaison_ET_PT_SINRH.csv")
OUT_DIR = os.path.join(OUTPUTS_DIR, "Toutes_Comparaisons", "PT_SINRH")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    if not os.path.exists(CSV_PT):
        print(f"❌ Le fichier {CSV_PT} n'existe pas. Lancez d'abord comparer_pt_sinrh.py")
        return
        
    df = pd.read_csv(CSV_PT)
    df['Date'] = pd.to_datetime(df['Date'])
    
    has_pure = 'ET_pure_ICOS (mm/h)' in df.columns
    
    sites = df['Site'].unique()
    print(f"📊 Génération des graphiques PT-SINRH pour {len(sites)} sites...")
    
    # Référence = TTME Landsat-ICOS (ET_pixel)
    var_et_ttme = 'ET_pixel (mm/h)'
    var_et_pt = 'ET_PT_SINRH (mm/h)'
    var_et_pure = 'ET_pure_ICOS (mm/h)' if has_pure else None
    
    for site in sorted(sites):
        df_site = df[df['Site'] == site].copy().sort_values('Date')
        
        if len(df_site) < 2:
            print(f"⚠️ Pas assez de données pour le site {site}. Ignoré.")
            continue
        
        et_ttme = df_site[var_et_ttme]
        et_pt = df_site[var_et_pt]
        et_pure = df_site[var_et_pure] if has_pure else pd.Series([np.nan]*len(df_site))
        
        # --- Métriques ---
        def calc_metrics(et_model, et_ref):
            mask = et_ref.notna() & et_model.notna()
            if mask.sum() >= 2:
                r = np.corrcoef(et_ref[mask], et_model[mask])[0, 1]
                r2 = r ** 2
                rmse = np.sqrt(mean_squared_error(et_ref[mask], et_model[mask]))
                bias = np.mean(et_model[mask] - et_ref[mask])
                return f"{r2:.3f}", f"{rmse:.3f}", f"{bias:.3f}"
            return "N/A", "N/A", "N/A"
        
        # Performances vs TTME Landsat
        r2_pt, rmse_pt, bias_pt = calc_metrics(et_pt, et_ttme)
        
        cell_text = [
            ["PT-SINRH vs TTME Landsat", r2_pt, rmse_pt, bias_pt],
        ]
        
        if has_pure:
            r2_pt_pure, rmse_pt_pure, bias_pt_pure = calc_metrics(et_pt, et_pure)
            cell_text.append(["PT-SINRH vs TTME Pure ICOS", r2_pt_pure, rmse_pt_pure, bias_pt_pure])
            
            r2_ttme, rmse_ttme, bias_ttme = calc_metrics(et_ttme, et_pure)
            cell_text.append(["TTME Landsat vs TTME Pure ICOS", r2_ttme, rmse_ttme, bias_ttme])
        
        # --- Figure ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Panel 1 : Tableau
        ax1 = axes[0]
        ax1.axis('tight')
        ax1.axis('off')
        
        col_labels = ["Comparaison", "r²", "RMSE (mm/h)", "Biais (mm/h)"]
        table = ax1.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
        table.scale(1, 2)
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#2d6a4f')
        
        ax1.set_title("Performances des modèles", fontsize=13, weight='bold', pad=20)
        
        # Panel 2 : Série temporelle
        ax2 = axes[1]
        
        # TTME Landsat
        valid_ttme = df_site.dropna(subset=[var_et_ttme])
        ax2.plot(valid_ttme['Date'], valid_ttme[var_et_ttme],
                 marker='s', linestyle='-', color='dodgerblue', label='TTME (Landsat)',
                 linewidth=1.5, alpha=0.9)
        
        # Pure ICOS (TTME)
        if has_pure:
            valid_pure = df_site.dropna(subset=[var_et_pure])
            ax2.plot(valid_pure['Date'], valid_pure[var_et_pure],
                     marker='D', linestyle='--', color='purple', label='TTME Pure ICOS',
                     linewidth=2, alpha=0.9)
        
        # PT-SINRH
        valid_pt = df_site.dropna(subset=[var_et_pt])
        ax2.plot(valid_pt['Date'], valid_pt[var_et_pt],
                 marker='^', linestyle='-', color='#2d6a4f', label='PT-SINRH',
                 linewidth=2, alpha=1.0, markersize=8)
        
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylabel("Évapotranspiration (mm/h)", fontsize=11)
        ax2.set_title("Évolution Temporelle", fontsize=13)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(fontsize=10)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.suptitle(f"Comparaison PT-SINRH — Site : {site}", fontsize=16, y=1.05, weight='bold')
        plt.tight_layout()
        
        out_png = os.path.join(OUT_DIR, f"Comparaison_PT_SINRH_{site}.png")
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close()
    
    # ===================================================================
    # Graphique Global : Performances agrégées
    # ===================================================================
    df_all = df.dropna(subset=[var_et_ttme, var_et_pt])
    if len(df_all) >= 2:
        et_all_ttme = df_all[var_et_ttme]
        et_all_pt = df_all[var_et_pt]
        
        r_glob = np.corrcoef(et_all_ttme, et_all_pt)[0, 1]
        rmse_glob = np.sqrt(mean_squared_error(et_all_ttme, et_all_pt))
        bias_glob = np.mean(et_all_pt - et_all_ttme)
        
        # Bar chart des MAE par site
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        site_metrics = []
        for site in sorted(df_all['Site'].unique()):
            ds = df_all[df_all['Site'] == site]
            if len(ds) >= 2:
                r_s = np.corrcoef(ds[var_et_ttme], ds[var_et_pt])[0, 1]
                rmse_s = np.sqrt(mean_squared_error(ds[var_et_ttme], ds[var_et_pt]))
                mae_s = np.mean(np.abs(ds[var_et_pt] - ds[var_et_ttme]))
                bias_s = np.mean(ds[var_et_pt] - ds[var_et_ttme])
                site_metrics.append({
                    'Site': site, 'r²': r_s**2, 'RMSE': rmse_s, 'MAE': mae_s, 'Biais': bias_s,
                    'N': len(ds)
                })
        
        df_metrics = pd.DataFrame(site_metrics)
        
        # Panel 1 : r² par site
        ax1 = axes[0]
        bars = ax1.bar(df_metrics['Site'], df_metrics['r²'], color='#2d6a4f', alpha=0.85)
        ax1.set_ylabel("r²", fontsize=12)
        ax1.set_title("Corrélation par site (PT-SINRH vs TTME)", fontsize=12, weight='bold')
        ax1.set_ylim(0, 1)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")
        for bar, val in zip(bars, df_metrics['r²']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha='center', fontsize=9)
        
        # Panel 2 : RMSE par site
        ax2 = axes[1]
        bars2 = ax2.bar(df_metrics['Site'], df_metrics['RMSE'], color='coral', alpha=0.85)
        ax2.set_ylabel("RMSE (mm/h)", fontsize=12)
        ax2.set_title("RMSE par site", fontsize=12, weight='bold')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")
        for bar, val in zip(bars2, df_metrics['RMSE']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha='center', fontsize=9)
        
        # Panel 3 : Boxplot des erreurs
        ax3 = axes[2]
        errors_by_site = [df_all[df_all['Site'] == s][var_et_pt].values - df_all[df_all['Site'] == s][var_et_ttme].values
                          for s in sorted(df_all['Site'].unique())]
        bp = ax3.boxplot(errors_by_site, labels=sorted(df_all['Site'].unique()),
                         patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#2d6a4f')
            patch.set_alpha(0.5)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax3.set_ylabel("Erreur (mm/h)", fontsize=12)
        ax3.set_title("Distribution des erreurs", fontsize=12, weight='bold')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")
        
        plt.suptitle(f"Performances Globales PT-SINRH (N={len(df_all)}, R²={r_glob**2:.3f}, RMSE={rmse_glob:.3f})",
                     fontsize=14, weight='bold')
        plt.tight_layout()
        
        out_glob = os.path.join(OUT_DIR, "Performances_PT_SINRH_Globales.png")
        plt.savefig(out_glob, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📈 Graphique global sauvegardé : {out_glob}")
    
    print(f"\n✅ Tous les graphiques PT-SINRH sauvegardés dans : {OUT_DIR}")


if __name__ == "__main__":
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
