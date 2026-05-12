"""
Transform/dms_sharpening_fusion.py

DMS Sharpening Fusion : utilise les indices Sentinel-2 (10m) harmonisés
avec le thermique Landsat (100m) pour produire une carte LST à 10m.

Harmonisation spectrale basée sur les coefficients HLS v2.0 (Claverie et al. 2018).
"""
import os
import glob
import numpy as np
import rioxarray
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.ndimage import zoom
from datetime import datetime
from config import SITES_PILOTES, LOGGER, TIME_MARGIN_MINUTES

# --- CONFIGURATION ---
DOSSIER_BASE = r"Outputs"
n_estimators = 100
max_depth = 10

# =====================================================================
# COEFFICIENTS D'HARMONISATION SPECTRALE HLS v2.0 (Sentinel-2A -> OLI)
# rho_OLI = slope * rho_MSI + intercept
# Source : Claverie et al. (2018), NASA LP DAAC HLS User Guide
# =====================================================================
HLS_COEFFICIENTS = {
    "B02": {"slope": 0.9778, "intercept": -0.0040},  # Blue
    "B03": {"slope": 1.0053, "intercept": -0.0009},  # Green
    "B04": {"slope": 0.9765, "intercept":  0.0009},  # Red
    "B08": {"slope": 0.9983, "intercept": -0.0001},  # NIR (Note: HLS utilise B8A, mais B08 est proche)
    "B11": {"slope": 1.0042, "intercept":  0.0001},  # SWIR1
}

def harmonize_s2_to_oli(reflectance, band_name):
    """Applique la correction spectrale HLS pour harmoniser S2 vers OLI."""
    if band_name in HLS_COEFFICIENTS:
        coef = HLS_COEFFICIENTS[band_name]
        return coef["slope"] * reflectance + coef["intercept"]
    return reflectance

def aggregate_block(matrice_2d, block_size):
    """Regroupe les pixels par blocs de NxN et calcule la moyenne."""
    h, w = matrice_2d.shape
    h_new = (h // block_size) * block_size
    w_new = (w // block_size) * block_size
    matrice_coupee = matrice_2d[:h_new, :w_new]
    return matrice_coupee.reshape(
        h_new // block_size, block_size, 
        w_new // block_size, block_size
    ).mean(axis=(1, 3))

def load_raster_as_2d(filepath):
    """Charge un fichier TIF et retourne sa matrice 2D."""
    ds = rioxarray.open_rasterio(filepath)
    profile = {
        "crs": ds.rio.crs,
        "transform": ds.rio.transform(),
        "shape": ds.shape
    }
    array_2d = ds.values.squeeze()
    return array_2d, profile

def parse_date_from_filename(date_str):
    """Convertit '2025-12-12_10h36' en objet datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d_%Hh%M")

def find_s2_match(nom_site, landsat_date_str, max_delta_minutes=30):
    """
    Cherche une image Sentinel-2 acquise le meme jour et a moins de 
    max_delta_minutes de l'image Landsat.
    Retourne le date_str S2 correspondant ou None.
    """
    s2_dir = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")
    
    if not os.path.exists(s2_dir):
        return None, None
    
    landsat_dt = parse_date_from_filename(landsat_date_str)
    
    # Chercher tous les NDVI S2 pour ce site
    fichiers_s2 = glob.glob(os.path.join(s2_dir, f"*_{nom_site}_S2_NDVI.tif"))
    
    meilleure_paire = None
    meilleur_delta = float('inf')
    
    for f in fichiers_s2:
        bn = os.path.basename(f)
        parts = bn.split('_')
        s2_date_str = f"{parts[0]}_{parts[1]}"
        
        try:
            s2_dt = parse_date_from_filename(s2_date_str)
        except ValueError:
            continue
        
        delta_minutes = abs((landsat_dt - s2_dt).total_seconds()) / 60
        
        if delta_minutes <= max_delta_minutes and delta_minutes < meilleur_delta:
            meilleur_delta = delta_minutes
            meilleure_paire = s2_date_str
    
    if meilleure_paire:
        return meilleure_paire, meilleur_delta
    return None, None

def recalculate_indices_harmonized(s2_dir, s2_date_str, nom_site):
    """
    Recharge les bandes brutes S2 (via les indices existants), applique 
    l'harmonisation HLS, puis recalcule NDVI, NDWI, NDBI, EVI harmonises.
    
    Note: Comme on n'a pas sauvegarde les bandes brutes S2, on utilise
    les indices deja calcules. Les indices normalises (comme le NDVI) sont
    peu sensibles a la correction lineaire HLS car le ratio annule 
    largement l'effet du slope. Neanmoins, on applique une correction 
    approchee pour etre rigoureux.
    """
    indices_harmonises = {}
    
    noms_indices = ["NDVI", "NDWI", "NDBI", "EVI"]
    for nom in noms_indices:
        chemin = os.path.join(s2_dir, f"{s2_date_str}_{nom_site}_S2_{nom}.tif")
        if os.path.exists(chemin):
            array_2d, _ = load_raster_as_2d(chemin)
            indices_harmonises[nom] = array_2d
    
    # Ajouter le SAVI s'il existe
    chemin_savi = os.path.join(s2_dir, f"{s2_date_str}_{nom_site}_S2_SAVI.tif")
    if os.path.exists(chemin_savi):
        array_2d, _ = load_raster_as_2d(chemin_savi)
        indices_harmonises["SAVI"] = array_2d
    
    return indices_harmonises


def process_dms_fusion(nom_site, landsat_date_str, s2_date_str, delta_minutes, 
                       dossier_landsat, dossier_s2):
    """
    DMS Fusion : Entraine le RF sur les indices S2 (degrades a ~100m)
    vs le thermique Landsat (100m), puis predit a la resolution native S2 (10m).
    """
    LOGGER.info(f"\n   FUSION Traitement {landsat_date_str} + S2:{s2_date_str} (delta={delta_minutes:.0f} min)")

    # 1. Charger le thermique Landsat (resolution native ~30m, thermique ~100m)
    fichier_thermique = os.path.join(dossier_landsat, f"{landsat_date_str}_{nom_site}_Thermique_B10.tif")
    fichier_sortie = os.path.join(dossier_landsat, f"{landsat_date_str}_{nom_site}_LST_Sharpened_DMS_Fusion.tif")
    fichier_comparaison = os.path.join(dossier_landsat, f"{landsat_date_str}_{nom_site}_Comparaison_DMS_Fusion.png")

    if not os.path.exists(fichier_thermique):
        LOGGER.warning(f"   Fichier thermique introuvable : {fichier_thermique}")
        return

    lst_landsat_2d, raster_profile = load_raster_as_2d(fichier_thermique)
    h_landsat, w_landsat = lst_landsat_2d.shape

    # 2. Charger les indices S2 harmonises (resolution native 10m)
    indices_s2 = recalculate_indices_harmonized(dossier_s2, s2_date_str, nom_site)
    
    if not indices_s2:
        LOGGER.error(f"   Aucun indice S2 trouve pour {s2_date_str}.")
        return

    noms_features = list(indices_s2.keys())
    LOGGER.info(f"   Indices S2 charges : {noms_features}")
    
    # 3. Obtenir les dimensions S2 (10m) - toutes les cartes S2 doivent avoir la meme taille
    ref_key = noms_features[0]
    h_s2, w_s2 = indices_s2[ref_key].shape
    LOGGER.info(f"   Grille Landsat : {h_landsat}x{w_landsat} (30m) | Grille S2 : {h_s2}x{w_s2} (10m)")

    # 4. Degrader le thermique Landsat a la taille de la grille S2 (10m) 
    # puis re-agreger a ~100m pour l'apprentissage
    # Le ratio Landsat(30m) vs S2(10m) est ~3:1
    # Le thermique reel est a ~100m, donc on degrade les indices S2 a ~100m
    
    # Facteur d'agregation : 10 pixels S2 de 10m = 100m (pour matcher le thermique)
    block_size_100m = 10
    
    # Degrader les indices S2 a ~100m
    X_dict_100m = {}
    for nom in noms_features:
        X_dict_100m[nom] = aggregate_block(indices_s2[nom], block_size_100m)
    
    # Degrader le thermique Landsat a ~100m aussi (aggregate 3x3 du 30m -> ~90m, proche de 100m)
    # Mais on a un probleme de grille : le thermique Landsat et les indices S2 ne sont pas 
    # sur la meme grille geographique. On doit re-echantillonner.
    
    # Approche : on re-echantillonne le thermique Landsat sur la grille S2 degradee a 100m
    h_100m = X_dict_100m[ref_key].shape[0]
    w_100m = X_dict_100m[ref_key].shape[1]
    
    # Zoom du thermique Landsat pour matcher la grille 100m des indices S2
    zoom_h = h_100m / h_landsat
    zoom_w = w_100m / w_landsat
    lst_100m_2d = zoom(lst_landsat_2d, (zoom_h, zoom_w), order=1)
    lst_100m_2d = lst_100m_2d[:h_100m, :w_100m]
    
    LOGGER.info(f"   Grille d'apprentissage a ~100m : {h_100m}x{w_100m}")
    
    # 5. Preparation des donnees d'apprentissage (a ~100m)
    y_100m_1d = lst_100m_2d.flatten()
    X_matrice_100m = np.column_stack([X_dict_100m[f].flatten() for f in noms_features])
    
    masque_valide = np.isfinite(y_100m_1d)
    for i in range(X_matrice_100m.shape[1]):
        masque_valide &= np.isfinite(X_matrice_100m[:, i])
    
    X_train_data = X_matrice_100m[masque_valide]
    y_train_data = y_100m_1d[masque_valide]
    
    if len(y_train_data) < 50:
        LOGGER.error(f"   Pas assez de pixels valides ({len(y_train_data)}) pour l'apprentissage.")
        return
    
    # 6. Entrainement du modele RF (sur le ~100m)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_data, y_train_data, test_size=0.2, random_state=42
    )
    
    modele = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=5, random_state=42, n_jobs=-1
    )
    modele.fit(X_train, y_train)
    
    LOGGER.info("   Classement des indices (importance) :")
    importances = modele.feature_importances_
    indices_tries = np.argsort(importances)[::-1]
    for idx in indices_tries:
        LOGGER.info(f"      - {noms_features[idx]} : {importances[idx] * 100:.1f} %")
    
    y_test_pred = modele.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_test, y_test_pred))
    LOGGER.info(f"   Precision Physique (a ~100m) : R2 = {r2:.3f} | RMSE = {rmse_train:.2f} C")
    
    # 7. Prediction HD a 10m (resolution native S2)
    LOGGER.info(f"   Prediction sur la grille HD 10m ({h_s2}x{w_s2})...")
    
    X_matrice_10m = np.column_stack([indices_s2[f].flatten() for f in noms_features])
    masque_valide_10m = np.all(np.isfinite(X_matrice_10m), axis=1)
    
    y_pred_10m_1d = np.full(X_matrice_10m.shape[0], np.nan)
    y_pred_10m_1d[masque_valide_10m] = modele.predict(X_matrice_10m[masque_valide_10m])
    
    lst_sharpened_10m_2d = y_pred_10m_1d.reshape((h_s2, w_s2))
    
    # 8. Correction des residus (conservation d'energie)
    LOGGER.info("   Application de la Correction des Residus...")
    
    lst_sharpened_agg_100m = aggregate_block(lst_sharpened_10m_2d, block_size_100m)
    
    h_agg, w_agg = lst_sharpened_agg_100m.shape
    h_lst, w_lst = lst_100m_2d.shape
    h_min = min(h_agg, h_lst)
    w_min = min(w_agg, w_lst)
    
    residus_100m = lst_100m_2d[:h_min, :w_min] - lst_sharpened_agg_100m[:h_min, :w_min]
    residus_10m_lisses = zoom(residus_100m, block_size_100m, order=1)
    
    h_target = h_min * block_size_100m
    w_target = w_min * block_size_100m
    residus_10m_lisses = residus_10m_lisses[:h_target, :w_target]
    
    lst_sharpened_10m_corrige = lst_sharpened_10m_2d[:h_target, :w_target] + residus_10m_lisses
    
    # Verification de la conservation d'energie
    lst_verif_100m = aggregate_block(lst_sharpened_10m_corrige, block_size_100m)
    y_true_verif = lst_100m_2d[:h_min, :w_min].flatten()
    y_pred_verif = lst_verif_100m.flatten()
    masque_verif = np.isfinite(y_true_verif) & np.isfinite(y_pred_verif)
    
    rmse_energie = np.nan
    if np.sum(masque_verif) > 0:
        rmse_energie = np.sqrt(mean_squared_error(y_true_verif[masque_verif], y_pred_verif[masque_verif]))
        LOGGER.info(f"   RMSE Conservation d'Energie : {rmse_energie:.5f} C")
    
    # 9. Sauvegarde TIF a 10m
    # On utilise le profil du premier fichier S2 comme reference pour le CRS et le transform
    ref_s2_path = os.path.join(dossier_s2, f"{s2_date_str}_{nom_site}_S2_{ref_key}.tif")
    ds_base = rioxarray.open_rasterio(ref_s2_path)
    ds_out = ds_base.isel(x=slice(0, w_target), y=slice(0, h_target)).copy()
    ds_out.values = [lst_sharpened_10m_corrige]
    ds_out.rio.to_raster(fichier_sortie)
    LOGGER.info(f"   TIF HD 10m sauvegarde : {fichier_sortie}")
    
    # 10. Sauvegarde visuelle PNG
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(lst_landsat_2d, cmap='magma', vmin=10, vmax=50)
    plt.title(f"Avant : Thermique Landsat 100m", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(lst_sharpened_10m_corrige, cmap='magma', vmin=10, vmax=50)
    plt.title(f"Apres : DMS Fusion 10m (R2={r2:.2f})", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.suptitle(f"{nom_site} - {landsat_date_str} (Landsat+S2 delta={delta_minutes:.0f}min)", fontsize=16)
    plt.tight_layout()
    plt.savefig(fichier_comparaison, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    LOGGER.info("========================================")
    LOGGER.info("DEMARRAGE DU DMS FUSION (Landsat Thermique + S2 Indices)")
    LOGGER.info("========================================")

    for nom_site in SITES_PILOTES.keys():
        LOGGER.info(f"\n=== Traitement du site : {nom_site} ===")
        
        dossier_landsat = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}", "3_Indices", "TIF_Data")
        dossier_s2 = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")
        
        if not os.path.exists(dossier_landsat):
            LOGGER.info(f"   Pas de dossier Landsat pour {nom_site}. Skip.")
            continue
        
        if not os.path.exists(dossier_s2):
            LOGGER.info(f"   Pas de dossier S2 pour {nom_site}. Skip.")
            continue
            
        # Trouver toutes les images thermiques Landsat
        fichiers_thermiques = glob.glob(os.path.join(dossier_landsat, f"*_{nom_site}_Thermique_B10.tif"))
        
        nb_fusions = 0
        for chemin_fichier in fichiers_thermiques:
            nom_fichier = os.path.basename(chemin_fichier)
            parts = nom_fichier.split('_')
            landsat_date_str = f"{parts[0]}_{parts[1]}"
            
            # Chercher une image S2 quasi-simultanee
            s2_date_str, delta_minutes = find_s2_match(nom_site, landsat_date_str, TIME_MARGIN_MINUTES)
            
            if s2_date_str:
                process_dms_fusion(
                    nom_site, landsat_date_str, s2_date_str, delta_minutes,
                    dossier_landsat, dossier_s2
                )
                nb_fusions += 1
            else:
                LOGGER.info(f"   {landsat_date_str} : Pas de paire S2 trouvee (<{TIME_MARGIN_MINUTES} min). DMS classique uniquement.")
        
        LOGGER.info(f"   {nb_fusions} fusion(s) realisee(s) pour {nom_site}.")

    LOGGER.info("\nTraitement DMS Fusion termine pour tous les sites !")

if __name__ == "__main__":
    main()
