import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pystac_client import Client
import planetary_computer
from config import LOGGER

# --- CONFIGURATION ---
CHEMIN_MANIFEST = os.path.join("Outputs", "manifest_extraction.json")
DOSSIER_STATS = os.path.join("Outputs", "Statistiques_Nuages")

def charger_donnees_stac(manifest_path):
    """Récupère les métadonnées STAC à partir des IDs du manifeste."""
    if not os.path.exists(manifest_path):
        LOGGER.error(f"❌ Manifeste introuvable : {manifest_path}")
        return []

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    donnees = []
    for site, stac_ids in manifest.items():
        if not stac_ids: continue
        LOGGER.info(f"   📥 Récupération des métadonnées : {site}")
        search = catalog.search(ids=stac_ids, collections=["landsat-c2-l2"])
        items = search.item_collection()

        for item in items:
            donnees.append({
                "Site": site,
                "Date": pd.to_datetime(item.datetime).tz_localize(None),
                "CloudCover": item.properties.get("eo:cloud_cover", 0)
            })
    return donnees

def generer_series_temporelles(df, site):
    """Génère un graphique de série temporelle pour un site spécifique."""
    df_site = df[df['Site'] == site].sort_index()

    plt.figure(figsize=(15, 7))
    
    # 1. Tracer les points d'observations réels
    plt.scatter(df_site.index, df_site['CloudCover'], color='blue', alpha=0.5, label='Observations Landsat')

    # 2. Tracer la moyenne mobile hebdomadaire (Tendance)
    # On resample par semaine ('W') et on fait la moyenne
    df_weekly = df_site['CloudCover'].resample('W').mean()
    plt.plot(df_weekly.index, df_weekly.values, color='navy', linewidth=2, label='Moyenne Hebdomadaire')

    # 3. Ajouter des zones de seuils colorées pour la lisibilité
    plt.axhspan(0, 30, color='green', alpha=0.1, label='Zone Optimale (<30%)')
    plt.axhspan(30, 70, color='orange', alpha=0.1, label='Zone Risquée (30-70%)')
    plt.axhspan(70, 100, color='red', alpha=0.1, label='Zone Inexploitable (>70%)')

    # Configuration du graphique
    plt.title(f"Évolution de la Couverture Nuageuse - {site}", fontsize=16)
    plt.ylabel("Pourcentage de Nuages (%)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.ylim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))

    # Sauvegarde
    path = os.path.join(DOSSIER_STATS, f"Serie_Temporelle_Nuages_{site}.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"   📈 Série temporelle générée : {path}")

def generer_histogramme_empile(df, site):
    """Génère l'histogramme des paliers de nuages (déjà demandé)."""
    df_site = df[df['Site'] == site].copy()
    bins = [0, 30, 60, 70, 100]
    labels = ["<30%", "30-60%", "60-70%", ">70%"]
    df_site['Cat'] = pd.cut(df_site['CloudCover'], bins=bins, labels=labels, include_lowest=True)
    
    stats = df_site.groupby([pd.Grouper(freq='ME'), 'Cat']).size().unstack(fill_value=0)
    stats.index = stats.index.strftime('%b %Y')

    ax = stats.plot(kind='bar', stacked=True, figsize=(12, 6), color=['#22c55e', '#facc15', '#f97316', '#ef4444'])
    plt.title(f"Répartition Mensuelle des Nuages - {site}")
    plt.ylabel("Nombre d'images")
    plt.tight_layout()
    
    path = os.path.join(DOSSIER_STATS, f"Histogramme_Nuages_{site}.png")
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    os.makedirs(DOSSIER_STATS, exist_ok=True)
    donnees = charger_donnees_stac(CHEMIN_MANIFEST)
    
    if not donnees: return
    
    df = pd.DataFrame(donnees).set_index('Date')
    
    for site in df['Site'].unique():
        LOGGER.info(f"📊 Analyse pour {site}...")
        generer_series_temporelles(df, site)
        generer_histogramme_empile(df, site)

if __name__ == "__main__":
    main()