"""
Transform/dms_sharpening_fusion.py

DMS Sharpening Fusion : utilise les indices Sentinel-2 (10m)
avec le thermique ECOSTRESS (~70m) pour produire une carte LST à 10m.

Démarche identique à dms_sharpening.py (Landsat) :
  1. Charger le thermique ECOSTRESS (~70m) et les indices S2 (10m)
  2. Dégrader les indices S2 à ~70m (block 7x7) pour l'apprentissage
  3. Entraîner un Random Forest à ~70m
  4. Prédire à la résolution native S2 (10m)
  5. Correction des résidus (conservation d'énergie)
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
n_estimators = 200
max_depth = 20

# Block size pour dégrader les indices S2 (10m) à la résolution ECOSTRESS (~70m)
# 7 pixels de 10m = 70m
BLOCK_SIZE_ECO = 7


def aggregate_block(matrice_2d, block_size):
    """Regroupe les pixels par blocs de NxN et calcule la moyenne."""
    h, w = matrice_2d.shape
    h_new = (h // block_size) * block_size
    w_new = (w // block_size) * block_size
    matrice_coupee = matrice_2d[:h_new, :w_new]
    # np.mean transmettra les NaN s'il y a un pixel nuageux S2 dans le bloc,
    # ce qui est souhaité pour rejeter le bloc complet de l'entraînement.
    return matrice_coupee.reshape(
        h_new // block_size, block_size, 
        w_new // block_size, block_size
    ).mean(axis=(1, 3))


def calculate_homogeneity_mask(X_dict_10m, threshold=0.20):
    """
    Évalue la variance interne des pixels 10m au sein de leur pixel parent de 70m.
    Retourne un masque booléen 2D à 70m : True = Homogène (pur), False = Hétérogène (mixte).
    """
    premiere_matrice = list(X_dict_10m.values())[0]
    h, w = premiere_matrice.shape
    block_size = BLOCK_SIZE_ECO
    h_new, w_new = (h // block_size) * block_size, (w // block_size) * block_size
    
    cv_total = np.zeros((h_new // block_size, w_new // block_size))
    nb_features = len(X_dict_10m)
    
    for _, matrice in X_dict_10m.items():
        matrice_coupee = matrice[:h_new, :w_new]
        blocks = matrice_coupee.reshape(h_new // block_size, block_size, w_new // block_size, block_size)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            std_blocks = np.nanstd(blocks, axis=(1, 3))
            mean_blocks = np.nanmean(blocks, axis=(1, 3))
            cv_blocks = np.where(mean_blocks != 0, std_blocks / np.abs(mean_blocks), 0)
            cv_blocks = np.nan_to_num(cv_blocks, nan=1.0)
        
        cv_total += cv_blocks
        
    cv_moyen = cv_total / nb_features
    masque_homogene_2d = cv_moyen <= threshold
    return masque_homogene_2d


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


def find_s2_match(nom_site, eco_date_str, max_delta_minutes=30):
    """
    Cherche une image Sentinel-2 acquise le même jour et à moins de 
    max_delta_minutes de l'image ECOSTRESS.
    Retourne le date_str S2 correspondant ou None.
    """
    s2_dir = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")
    
    if not os.path.exists(s2_dir):
        return None, None
    
    eco_dt = parse_date_from_filename(eco_date_str)
    
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
        
        delta_minutes = abs((eco_dt - s2_dt).total_seconds()) / 60
        
        if delta_minutes <= max_delta_minutes and delta_minutes < meilleur_delta:
            meilleur_delta = delta_minutes
            meilleure_paire = s2_date_str
    
    if meilleure_paire:
        return meilleure_paire, meilleur_delta
    return None, None


def load_s2_indices(dossier_s2, s2_date_str, nom_site):
    """
    Charge les indices Sentinel-2 (NDVI, NDWI, NDBI, EVI, SAVI)
    directement depuis les fichiers TIF sans harmonisation HLS.
    """
    indices = {}
    
    noms_indices = ["NDVI", "NDWI", "NDBI", "EVI", "SAVI"]
    for nom in noms_indices:
        chemin = os.path.join(dossier_s2, f"{s2_date_str}_{nom_site}_S2_{nom}.tif")
        if os.path.exists(chemin):
            array_2d, _ = load_raster_as_2d(chemin)
            indices[nom] = array_2d
    
    return indices


def process_dms_fusion(nom_site, eco_date_str, s2_date_str, delta_minutes, 
                       dossier_ecostress, dossier_s2):
    """
    DMS Fusion : Entraîne le RF sur les indices S2 (dégradés à ~70m)
    vs le thermique ECOSTRESS (~70m), puis prédit à la résolution native S2 (10m).
    """
    LOGGER.info(f"\n   FUSION DMS : ECOSTRESS {eco_date_str} + S2:{s2_date_str} (delta={delta_minutes:.0f} min)")

    # 1. Charger le thermique ECOSTRESS (~70m)
    fichier_thermique = os.path.join(dossier_ecostress, f"{eco_date_str}_{nom_site}_ECOSTRESS_LST.tif")
    fichier_sortie = os.path.join(dossier_ecostress, f"{eco_date_str}_{nom_site}_LST_Sharpened_DMS_Fusion.tif")
    fichier_comparaison = os.path.join(dossier_ecostress, f"{eco_date_str}_{nom_site}_Comparaison_DMS_Fusion.png")

    if not os.path.exists(fichier_thermique):
        LOGGER.warning(f"   Fichier thermique ECOSTRESS introuvable : {fichier_thermique}")
        return

    lst_eco_2d, raster_profile = load_raster_as_2d(fichier_thermique)
    h_eco, w_eco = lst_eco_2d.shape

    # Vérifier que l'image ECOSTRESS a assez de pixels valides
    nb_valid_eco = np.count_nonzero(~np.isnan(lst_eco_2d) & (lst_eco_2d > -50) & (lst_eco_2d < 80))
    if nb_valid_eco < 50:
        LOGGER.warning(f"   Pas assez de pixels ECOSTRESS valides ({nb_valid_eco}). Skip.")
        return

    # 2. Charger les indices S2 (résolution native 10m) - PAS d'harmonisation HLS
    indices_s2 = load_s2_indices(dossier_s2, s2_date_str, nom_site)
    
    if not indices_s2:
        LOGGER.error(f"   Aucun indice S2 trouvé pour {s2_date_str}.")
        return

    noms_features = list(indices_s2.keys())
    LOGGER.info(f"   Indices S2 chargés : {noms_features}")
    
    # 3. Obtenir les dimensions S2 (10m)
    ref_key = noms_features[0]
    h_s2, w_s2 = indices_s2[ref_key].shape
    LOGGER.info(f"   Grille ECOSTRESS : {h_eco}x{w_eco} (~70m) | Grille S2 : {h_s2}x{w_s2} (10m)")

    # 4. Dégrader les indices S2 à ~70m pour l'apprentissage
    # 7 pixels S2 de 10m = 70m (pour matcher la résolution ECOSTRESS)
    X_dict_70m = {}
    for nom in noms_features:
        X_dict_70m[nom] = aggregate_block(indices_s2[nom], BLOCK_SIZE_ECO)
    
    # Dimensions de la grille dégradée à ~70m
    h_70m = X_dict_70m[ref_key].shape[0]
    w_70m = X_dict_70m[ref_key].shape[1]
    
    # Rééchantillonner le thermique ECOSTRESS pour matcher la grille 70m des indices S2
    zoom_h = h_70m / h_eco
    zoom_w = w_70m / w_eco
    lst_70m_2d = zoom(lst_eco_2d, (zoom_h, zoom_w), order=1)
    lst_70m_2d = lst_70m_2d[:h_70m, :w_70m]
    
    LOGGER.info(f"   Grille d'apprentissage à ~70m : {h_70m}x{w_70m}")
    
    # Calcul du masque d'homogénéité (rejet des pixels 70m trop hétérogènes)
    masque_homogene_2d = calculate_homogeneity_mask(indices_s2, threshold=0.20)
    masque_homogene_1d = masque_homogene_2d.flatten()
    
    # 5. Préparation des données d'apprentissage (à ~70m)
    y_70m_1d = lst_70m_2d.flatten()
    X_matrice_70m = np.column_stack([X_dict_70m[f].flatten() for f in noms_features])
    
    masque_valide = np.isfinite(y_70m_1d) & (y_70m_1d > -50) & (y_70m_1d < 80) & masque_homogene_1d
    for i in range(X_matrice_70m.shape[1]):
        masque_valide &= np.isfinite(X_matrice_70m[:, i])
    
    X_train_data = X_matrice_70m[masque_valide]
    y_train_data = y_70m_1d[masque_valide]
    
    if len(y_train_data) < 50:
        LOGGER.error(f"   Pas assez de pixels valides ({len(y_train_data)}) pour l'apprentissage.")
        return
    
    # 6. Entraînement du modèle RF (sur le ~70m)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_data, y_train_data, test_size=0.2, random_state=42
    )
    
    modele = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_leaf=5, random_state=42, n_jobs=-1,
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
    LOGGER.info(f"   Précision Physique (à ~70m) : R\u00b2 = {r2:.3f} | RMSE = {rmse_train:.2f} °C")
    
    # 7. Prédiction HD à 10m (résolution native S2)
    LOGGER.info(f"   Prédiction sur la grille HD 10m ({h_s2}x{w_s2})...")
    
    X_matrice_10m = np.column_stack([indices_s2[f].flatten() for f in noms_features])
    masque_valide_10m = np.all(np.isfinite(X_matrice_10m), axis=1)
    
    y_pred_10m_1d = np.full(X_matrice_10m.shape[0], np.nan)
    y_pred_10m_1d[masque_valide_10m] = modele.predict(X_matrice_10m[masque_valide_10m])
    
    lst_sharpened_10m_2d = y_pred_10m_1d.reshape((h_s2, w_s2))
    
    # Sauvegarder la version AVANT correction des résidus pour comparaison
    lst_sharpened_avant_correction = lst_sharpened_10m_2d.copy()
    
    # 8. Correction des résidus (conservation d'énergie)
    LOGGER.info("   Application de la Correction des Résidus...")
    
    lst_sharpened_agg_70m = aggregate_block(lst_sharpened_10m_2d, BLOCK_SIZE_ECO)
    
    h_agg, w_agg = lst_sharpened_agg_70m.shape
    h_lst, w_lst = lst_70m_2d.shape
    h_min = min(h_agg, h_lst)
    w_min = min(w_agg, w_lst)
    
    residus_70m = lst_70m_2d[:h_min, :w_min] - lst_sharpened_agg_70m[:h_min, :w_min]
    residus_10m_lisses = zoom(residus_70m, BLOCK_SIZE_ECO, order=1)
    
    h_target = h_min * BLOCK_SIZE_ECO
    w_target = w_min * BLOCK_SIZE_ECO
    residus_10m_lisses = residus_10m_lisses[:h_target, :w_target]
    
    lst_sharpened_10m_corrige = lst_sharpened_10m_2d[:h_target, :w_target] + residus_10m_lisses
    
    # Vérification de la conservation d'énergie
    lst_verif_70m = aggregate_block(lst_sharpened_10m_corrige, BLOCK_SIZE_ECO)
    y_true_verif = lst_70m_2d[:h_min, :w_min].flatten()
    y_pred_verif = lst_verif_70m.flatten()
    masque_verif = np.isfinite(y_true_verif) & np.isfinite(y_pred_verif)
    
    rmse_energie = np.nan
    if np.sum(masque_verif) > 0:
        rmse_energie = np.sqrt(mean_squared_error(y_true_verif[masque_verif], y_pred_verif[masque_verif]))
        LOGGER.info(f"   RMSE Conservation d'Énergie : {rmse_energie:.5f} °C")
    
    # 9. Sauvegarde TIF à 10m
    # On utilise le profil du premier fichier S2 comme référence pour le CRS et le transform
    ref_s2_path = os.path.join(dossier_s2, f"{s2_date_str}_{nom_site}_S2_{ref_key}.tif")
    ds_base = rioxarray.open_rasterio(ref_s2_path)
    ds_out = ds_base.isel(x=slice(0, w_target), y=slice(0, h_target)).copy()
    ds_out.values = [lst_sharpened_10m_corrige]
    ds_out.rio.to_raster(fichier_sortie)
    LOGGER.info(f"   TIF HD 10m sauvegardé : {fichier_sortie}")
    
    # 10. Sauvegarde visuelle PNG - Comparaison 3 panneaux
    lst_avant_affichage = lst_sharpened_avant_correction[:h_target, :w_target]
    masque_nan_avant = np.isnan(lst_sharpened_10m_corrige)
    lst_avant_affichage[masque_nan_avant] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    
    # Panneau 1 : Thermique ECOSTRESS original
    im0 = axes[0].imshow(lst_eco_2d, cmap='magma', vmin=np.nanpercentile(lst_eco_2d, 2), vmax=np.nanpercentile(lst_eco_2d, 98))
    axes[0].set_title("ECOSTRESS LST ~70m (original)", fontsize=13)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axis('off')
    
    # Panneau 2 : DMS Fusion SANS résidus
    im1 = axes[1].imshow(lst_avant_affichage, cmap='magma', vmin=np.nanpercentile(lst_eco_2d, 2), vmax=np.nanpercentile(lst_eco_2d, 98))
    axes[1].set_title(f"DMS Fusion sans r\u00e9sidus (R\u00b2={r2:.2f})", fontsize=13)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')
    
    # Panneau 3 : DMS Fusion AVEC résidus
    im2 = axes[2].imshow(lst_sharpened_10m_corrige, cmap='magma', vmin=np.nanpercentile(lst_eco_2d, 2), vmax=np.nanpercentile(lst_eco_2d, 98))
    axes[2].set_title(f"DMS Fusion avec résidus (RMSE={rmse_energie:.2f}°C)", fontsize=13)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')
    
    fig.suptitle(f"{nom_site} – {eco_date_str} (ECOSTRESS+S2 delta={delta_minutes:.0f}min) | Comparaison DMS Fusion", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fichier_comparaison, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    LOGGER.info("========================================")
    LOGGER.info("DÉMARRAGE DU DMS FUSION (ECOSTRESS Thermique + S2 Indices)")
    LOGGER.info("========================================")

    for nom_site in SITES_PILOTES.keys():
        LOGGER.info(f"\n=== Traitement du site : {nom_site} ===")
        
        dossier_ecostress = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}_ECOSTRESS", "TIF_Data")
        dossier_s2 = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")
        
        if not os.path.exists(dossier_ecostress):
            LOGGER.info(f"   Pas de dossier ECOSTRESS pour {nom_site}. Skip.")
            continue
        
        if not os.path.exists(dossier_s2):
            LOGGER.info(f"   Pas de dossier S2 pour {nom_site}. Skip.")
            continue
            
        # Trouver toutes les images ECOSTRESS LST
        fichiers_eco = glob.glob(os.path.join(dossier_ecostress, f"*_{nom_site}_ECOSTRESS_LST.tif"))
        
        LOGGER.info(f"   {len(fichiers_eco)} images ECOSTRESS détectées.")
        
        nb_fusions = 0
        for chemin_fichier in fichiers_eco:
            nom_fichier = os.path.basename(chemin_fichier)
            # Format : 2025-02-12_10h31_Gebesee_ECOSTRESS_LST.tif
            parts = nom_fichier.split('_')
            eco_date_str = f"{parts[0]}_{parts[1]}"
            
            # Chercher une image S2 quasi-simultanée
            s2_date_str, delta_minutes = find_s2_match(nom_site, eco_date_str, TIME_MARGIN_MINUTES)
            
            if s2_date_str:
                process_dms_fusion(
                    nom_site, eco_date_str, s2_date_str, delta_minutes,
                    dossier_ecostress, dossier_s2
                )
                nb_fusions += 1
            else:
                LOGGER.info(f"   {eco_date_str} : Pas de paire S2 trouvée (<{TIME_MARGIN_MINUTES} min).")
        
        LOGGER.info(f"   {nb_fusions} fusion(s) réalisée(s) pour {nom_site}.")

    LOGGER.info("\nTraitement DMS Fusion (ECOSTRESS+S2) terminé pour tous les sites !")

if __name__ == "__main__":
    main()
