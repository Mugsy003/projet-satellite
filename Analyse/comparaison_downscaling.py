"""
Analyse/comparaison_downscaling.py
====================================
Compare les performances du downscaling de Ta ERA5-Land :
- Ta ERA5 brute (scalaire, ~10.5km)
- Ta ERA5 downscalée (1050m, via stepwise RF)
- Ta ICOS (mesure terrain, référence)

Produit des graphiques MAE/RMSE et un scatter plot.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import rioxarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pyproj import Transformer

sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOGGER, SITES_PILOTES


# =============================================================================
# CONFIGURATION
# =============================================================================

SITES_TEST = ['Italy', 'Greece']
DATES_PAR_SITE = {
    'Italy': ['2023-06-02', '2023-06-17', '2023-06-25'],
    'Greece': ['2023-06-04', '2023-06-20', '2023-06-28'],
}

OUTPUT_DIR = "Outputs_Downscaling"


def get_ta_icos(site, date_str):
    """Récupère la Ta mesurée par ICOS pour un site/date (~10h30 UTC)."""
    csv_path = os.path.join("Outputs_ICOS", f"donnees_icos_{site}.csv")
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path, parse_dates=['TIMESTAMP'])
    target = pd.to_datetime(f"{date_str} 10:30:00")
    
    # Fenêtre de ±1h
    mask = (df['TIMESTAMP'] >= target - pd.Timedelta(hours=1)) & \
           (df['TIMESTAMP'] <= target + pd.Timedelta(hours=1))
    df_window = df[mask]
    
    if df_window.empty:
        return None
    
    # Trouver la ligne la plus proche
    idx = (df_window['TIMESTAMP'] - target).abs().idxmin()
    ta = df_window.loc[idx, 'TA_Consolide']
    return float(ta) if pd.notna(ta) else None


def get_ta_era5_brute(site, date_str):
    """Récupère la Ta ERA5 brute (scalaire) pour un site/date (~10h30 UTC)."""
    csv_path = os.path.join("Outputs_ERA5", f"donnees_era5_{site}.csv")
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path, parse_dates=['TIMESTAMP'])
    target = pd.to_datetime(f"{date_str} 10:30:00")
    
    mask = (df['TIMESTAMP'] >= target - pd.Timedelta(hours=1)) & \
           (df['TIMESTAMP'] <= target + pd.Timedelta(hours=1))
    df_window = df[mask]
    
    if df_window.empty:
        return None
    
    idx = (df_window['TIMESTAMP'] - target).abs().idxmin()
    ta = df_window.loc[idx, 'Ta (°C)']
    return float(ta) if pd.notna(ta) else None


def get_ta_downscaled(site, date_str):
    """Récupère la Ta downscalée au pixel le plus proche du site."""
    tif_path = os.path.join(OUTPUT_DIR, f"Ta_downscaled_{site}_{date_str}_1050m.tif")
    if not os.path.exists(tif_path):
        return None
    
    ds = rioxarray.open_rasterio(tif_path).squeeze()
    
    if site in SITES_PILOTES:
        coords = SITES_PILOTES[site]
        crs_raster = ds.rio.crs
        transformer = Transformer.from_crs("EPSG:4326", crs_raster, always_xy=True)
        x_utm, y_utm = transformer.transform(coords['lon'], coords['lat'])
    else:
        # Fallback to the center of the TIF
        x_utm = ds.x.values[len(ds.x) // 2]
        y_utm = ds.y.values[len(ds.y) // 2]
    
    # Extraire la valeur au pixel le plus proche
    try:
        val = ds.sel(x=x_utm, y=y_utm, method='nearest').values
        return float(val) if np.isfinite(val) else None
    except Exception:
        return None


def main():
    LOGGER.info("=" * 60)
    LOGGER.info("📊 COMPARAISON DOWNSCALING Ta ERA5-Land")
    LOGGER.info("=" * 60)
    
    # 1. Lancer le downscaling pour toutes les dates
    from Traitement.downscaling_Ta import process_site_date
    
    results = []
    
    for site in SITES_TEST:
        dates = DATES_PAR_SITE.get(site, [])
        for date_str in dates:
            LOGGER.info(f"\n{'─'*50}")
            LOGGER.info(f"📍 {site} — {date_str}")
            
            # Lancer le downscaling si pas déjà fait
            tif_path = os.path.join(OUTPUT_DIR, f"Ta_downscaled_{site}_{date_str}_1050m.tif")
            if not os.path.exists(tif_path):
                try:
                    process_site_date(site, date_str)
                except Exception as e:
                    LOGGER.error(f"   ❌ Erreur downscaling : {e}")
                    continue
            
            # Récupérer les 3 valeurs de Ta
            ta_icos = get_ta_icos(site, date_str)
            ta_era5 = get_ta_era5_brute(site, date_str)
            ta_ds = get_ta_downscaled(site, date_str)
            
            LOGGER.info(f"   Ta ICOS (ref)  : {ta_icos:.1f}°C" if ta_icos else "   Ta ICOS : N/A")
            LOGGER.info(f"   Ta ERA5 brute  : {ta_era5:.1f}°C" if ta_era5 else "   Ta ERA5 : N/A")
            LOGGER.info(f"   Ta downscalée  : {ta_ds:.1f}°C" if ta_ds else "   Ta DS   : N/A")
            
            results.append({
                'site': site,
                'date': date_str,
                'Ta_ICOS': ta_icos,
                'Ta_ERA5_brute': ta_era5,
                'Ta_downscaled': ta_ds,
            })
    
    if not results:
        LOGGER.error("❌ Aucun résultat à comparer.")
        return
    
    df = pd.DataFrame(results)
    
    # Calculer les erreurs
    df['Erreur_ERA5'] = df['Ta_ERA5_brute'] - df['Ta_ICOS']
    df['Erreur_DS'] = df['Ta_downscaled'] - df['Ta_ICOS']
    
    # Métriques globales
    valid_era5 = df.dropna(subset=['Erreur_ERA5'])
    valid_ds = df.dropna(subset=['Erreur_DS'])
    
    mae_era5 = valid_era5['Erreur_ERA5'].abs().mean()
    rmse_era5 = np.sqrt((valid_era5['Erreur_ERA5']**2).mean())
    bias_era5 = valid_era5['Erreur_ERA5'].mean()
    
    mae_ds = valid_ds['Erreur_DS'].abs().mean()
    rmse_ds = np.sqrt((valid_ds['Erreur_DS']**2).mean())
    bias_ds = valid_ds['Erreur_DS'].mean()
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("📊 RÉSULTATS GLOBAUX")
    LOGGER.info(f"{'='*60}")
    LOGGER.info(f"   ERA5 brute   : MAE={mae_era5:.2f}°C, RMSE={rmse_era5:.2f}°C, Biais={bias_era5:+.2f}°C")
    LOGGER.info(f"   Downscalée   : MAE={mae_ds:.2f}°C, RMSE={rmse_ds:.2f}°C, Biais={bias_ds:+.2f}°C")
    
    # Sauvegarder le tableau
    csv_path = os.path.join(OUTPUT_DIR, "comparaison_downscaling.csv")
    df.to_csv(csv_path, index=False)
    LOGGER.info(f"   💾 Tableau sauvegardé : {csv_path}")
    
    # --- GRAPHIQUES ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Comparaison Ta ERA5 brute vs Downscalée (référence ICOS)", 
                 fontsize=14, fontweight='bold')
    
    # 1. Scatter ERA5 brute vs ICOS
    ax = axes[0]
    valid = df.dropna(subset=['Ta_ERA5_brute', 'Ta_ICOS'])
    ax.scatter(valid['Ta_ICOS'], valid['Ta_ERA5_brute'], c='tab:blue', alpha=0.7, s=60, edgecolors='white')
    lims = [min(valid['Ta_ICOS'].min(), valid['Ta_ERA5_brute'].min()) - 2,
            max(valid['Ta_ICOS'].max(), valid['Ta_ERA5_brute'].max()) + 2]
    if not np.isnan(lims).any():
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    
    ax.set_xlabel('Ta ICOS mesurée (°C)', fontweight='bold')
    ax.set_ylabel("Ta ERA5 brute (°C)")
    ax.set_title(f"ERA5 brute\nMAE={mae_era5:.2f}°C, RMSE={rmse_era5:.2f}°C")
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    
    # 2. Scatter Downscalée vs ICOS
    ax = axes[1]
    valid = df.dropna(subset=['Ta_downscaled', 'Ta_ICOS'])
    ax.scatter(valid['Ta_ICOS'], valid['Ta_downscaled'], c='tab:orange', alpha=0.7, s=60, edgecolors='white')
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel("Ta ICOS (°C)")
    ax.set_ylabel("Ta Downscalée 1km (°C)")
    ax.set_title(f"Downscalée (1km)\nMAE={mae_ds:.2f}°C, RMSE={rmse_ds:.2f}°C")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    
    # 3. Barplot MAE/RMSE par site
    ax = axes[2]
    sites_list = df['site'].unique()
    x = np.arange(len(sites_list))
    width = 0.35
    
    mae_era5_site = [df[df['site']==s]['Erreur_ERA5'].abs().mean() for s in sites_list]
    mae_ds_site = [df[df['site']==s]['Erreur_DS'].dropna().abs().mean() for s in sites_list]
    
    bars1 = ax.bar(x - width/2, mae_era5_site, width, label='ERA5 brute', color='tab:blue', alpha=0.8)
    bars2 = ax.bar(x + width/2, mae_ds_site, width, label='Downscalée 1km', color='tab:orange', alpha=0.8)
    ax.set_ylabel("MAE (°C)")
    ax.set_title("MAE par site")
    ax.set_xticks(x)
    ax.set_xticklabels(sites_list, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "Comparaison_Downscaling_Ta.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"   📊 Graphique sauvegardé : {fig_path}")
    
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("✅ COMPARAISON TERMINÉE.")
    LOGGER.info(f"{'='*60}")


if __name__ == "__main__":
    main()
