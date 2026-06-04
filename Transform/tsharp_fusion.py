"""
Transform/tsharp_fusion.py

TsHARP Fusion : utilise le NDVI Sentinel-2 (10m)
avec le thermique ECOSTRESS (~70m) pour produire une carte LST à 10m.

Relation quadratique classique : LST = a * NDVI² + b * NDVI + c

Démarche identique à tsharp.py (Landsat) :
  1. Charger le thermique ECOSTRESS (~70m) et le NDVI S2 (10m)
  2. Dégrader le NDVI S2 à ~70m (block 7x7) pour l'apprentissage
  3. Fit quadratique à ~70m
  4. Prédire à la résolution native S2 (10m)
  5. Correction des résidus (conservation d'énergie)
"""

import os
import glob
import numpy as np
from scipy.ndimage import zoom
import rioxarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from config import SITES_PILOTES, LOGGER, TIME_MARGIN_MINUTES

DOSSIER_BASE = r"Outputs"

# Block size pour dégrader le NDVI S2 (10m) à la résolution ECOSTRESS (~70m)
BLOCK_SIZE_ECO = 7


# =====================================================================
# FONCTIONS TSHARP CORE
# =====================================================================

def fit_tsharp(coarse_lst, coarse_ndvi, mask=None):
    """Fit quadratique LST = a*NDVI^2 + b*NDVI + c."""
    lst_flat = coarse_lst.ravel().astype(np.float64)
    ndvi_flat = coarse_ndvi.ravel().astype(np.float64)

    if mask is not None:
        mask_flat = mask.ravel().astype(bool)
        lst_flat = lst_flat[mask_flat]
        ndvi_flat = ndvi_flat[mask_flat]

    design = np.column_stack([ndvi_flat**2, ndvi_flat, np.ones_like(ndvi_flat)])
    coefficients, _, _, _ = np.linalg.lstsq(design, lst_flat, rcond=None)

    return coefficients.astype(np.float64)


def predict_tsharp(coarse_lst, coarse_ndvi, fine_ndvi, coefficients=None, mask=None):
    """Prédit la LST fine résolution via TsHARP.
    Retourne (prediction_avec_residus, prediction_sans_residus)."""
    if coefficients is None:
        coefficients = fit_tsharp(coarse_lst, coarse_ndvi, mask=mask)

    a, b, c = coefficients

    coarse_predicted = (
        a * coarse_ndvi.astype(np.float64) ** 2
        + b * coarse_ndvi.astype(np.float64)
        + c
    )

    residual = coarse_lst.astype(np.float64) - coarse_predicted

    zoom_factors = (
        fine_ndvi.shape[0] / coarse_lst.shape[0],
        fine_ndvi.shape[1] / coarse_lst.shape[1],
    )
    residual_interp = zoom(residual, zoom_factors, order=1)

    # Prédiction sans résidus
    fine_predicted_no_residual = (
        a * fine_ndvi.astype(np.float64) ** 2
        + b * fine_ndvi.astype(np.float64)
        + c
    )

    # Prédiction avec résidus
    fine_predicted = fine_predicted_no_residual + residual_interp

    return fine_predicted.astype(np.float32), fine_predicted_no_residual.astype(np.float32)


# =====================================================================
# FONCTIONS UTILITAIRES
# =====================================================================

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

def calculate_homogeneity_mask(ndvi_10m, threshold=0.20):
    """
    Évalue la variance interne du NDVI S2 (10m) au sein de son pixel parent ECOSTRESS (~70m).
    Retourne un masque booléen 2D à 70m : True = Homogène (pur), False = Hétérogène (mixte).
    """
    h, w = ndvi_10m.shape
    block_size = BLOCK_SIZE_ECO
    h_new, w_new = (h // block_size) * block_size, (w // block_size) * block_size
    
    matrice_coupee = ndvi_10m[:h_new, :w_new]
    blocks = matrice_coupee.reshape(h_new // block_size, block_size, w_new // block_size, block_size)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        std_blocks = np.nanstd(blocks, axis=(1, 3))
        mean_blocks = np.nanmean(blocks, axis=(1, 3))
        cv_blocks = np.where(mean_blocks != 0, std_blocks / np.abs(mean_blocks), 0)
        cv_blocks = np.nan_to_num(cv_blocks, nan=1.0)
    
    masque_homogene_2d = cv_blocks <= threshold
    return masque_homogene_2d

def load_raster_as_2d(filepath):
    """Charge un fichier TIF et retourne sa matrice 2D."""
    ds = rioxarray.open_rasterio(filepath)
    array_2d = ds.values.squeeze()
    return array_2d

def parse_date_from_filename(date_str):
    """Convertit '2025-12-12_10h36' en objet datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d_%Hh%M")

def find_s2_match(nom_site, eco_date_str, max_delta_minutes=30):
    """Cherche une image S2 quasi-simultanée à une image ECOSTRESS."""
    s2_dir = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")
    
    if not os.path.exists(s2_dir):
        return None, None
    
    eco_dt = parse_date_from_filename(eco_date_str)
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


# =====================================================================
# FONCTION PRINCIPALE DE FUSION TSHARP
# =====================================================================

def process_tsharp_fusion(nom_site, eco_date_str, s2_date_str, delta_minutes,
                          dossier_ecostress, dossier_s2):
    """
    TsHARP Fusion : utilise le NDVI S2 (10m) avec le thermique ECOSTRESS (~70m).
    Apprentissage à ~70m, prédiction à 10m.
    """
    LOGGER.info(f"\n   FUSION TsHARP : ECOSTRESS {eco_date_str} + S2:{s2_date_str} (delta={delta_minutes:.0f} min)")

    # 1. Charger le thermique ECOSTRESS (~70m)
    fichier_thermique = os.path.join(dossier_ecostress, f"{eco_date_str}_{nom_site}_ECOSTRESS_LST.tif")
    fichier_sortie = os.path.join(dossier_ecostress, f"{eco_date_str}_{nom_site}_LST_Sharpened_TsHARP_Fusion.tif")
    fichier_comparaison = os.path.join(dossier_ecostress, f"{eco_date_str}_{nom_site}_Comparaison_TsHARP_Fusion.png")

    if not os.path.exists(fichier_thermique):
        LOGGER.warning(f"   Fichier thermique ECOSTRESS introuvable : {fichier_thermique}")
        return

    lst_eco_2d = load_raster_as_2d(fichier_thermique)
    h_eco, w_eco = lst_eco_2d.shape

    # Vérifier que l'image ECOSTRESS a assez de pixels valides
    nb_valid_eco = np.count_nonzero(~np.isnan(lst_eco_2d) & (lst_eco_2d > -50) & (lst_eco_2d < 80))
    if nb_valid_eco < 50:
        LOGGER.warning(f"   Pas assez de pixels ECOSTRESS valides ({nb_valid_eco}). Skip.")
        return

    # 2. Charger le NDVI S2 (10m) - PAS d'harmonisation HLS
    fichier_ndvi_s2 = os.path.join(dossier_s2, f"{s2_date_str}_{nom_site}_S2_NDVI.tif")
    
    if not os.path.exists(fichier_ndvi_s2):
        LOGGER.error(f"   NDVI S2 introuvable : {fichier_ndvi_s2}")
        return
    
    ndvi_s2_10m = load_raster_as_2d(fichier_ndvi_s2)
    
    h_s2, w_s2 = ndvi_s2_10m.shape
    LOGGER.info(f"   Grille ECOSTRESS : {h_eco}x{w_eco} (~70m) | Grille S2 : {h_s2}x{w_s2} (10m)")

    # 3. Dégrader tout à ~70m pour l'apprentissage
    # 7 pixels S2 de 10m = 70m (pour matcher la résolution ECOSTRESS)
    ndvi_70m = aggregate_block(ndvi_s2_10m, BLOCK_SIZE_ECO)
    h_70m, w_70m = ndvi_70m.shape
    
    # Calcul du masque d'homogénéité (rejet des pixels 70m trop hétérogènes)
    masque_homogene_2d = calculate_homogeneity_mask(ndvi_s2_10m, threshold=0.20)
    
    # Rééchantillonner le thermique ECOSTRESS sur la grille 70m
    zoom_h = h_70m / h_eco
    zoom_w = w_70m / w_eco
    lst_70m_2d = zoom(lst_eco_2d, (zoom_h, zoom_w), order=1)
    lst_70m_2d = lst_70m_2d[:h_70m, :w_70m]
    
    LOGGER.info(f"   Grille d'apprentissage à ~70m : {h_70m}x{w_70m}")

    # 4. TsHARP : apprentissage quadratique à ~70m, prédiction à 10m
    mask_70m = np.isfinite(lst_70m_2d) & np.isfinite(ndvi_70m) & (lst_70m_2d > -50) & (lst_70m_2d < 80) & masque_homogene_2d

    try:
        lst_sharpened_10m, lst_sans_residus_10m = predict_tsharp(
            coarse_lst=np.nan_to_num(lst_70m_2d, nan=np.nanmean(lst_70m_2d)),
            coarse_ndvi=np.nan_to_num(ndvi_70m, nan=np.nanmean(ndvi_70m)),
            fine_ndvi=np.nan_to_num(ndvi_s2_10m, nan=np.nanmean(ndvi_s2_10m)),
            mask=mask_70m
        )
    except Exception as e:
        LOGGER.error(f"   Erreur lors de TsHARP Fusion : {e}")
        return

    # 5. Ajuster la taille et remettre les NaN
    lst_sharpened_10m = lst_sharpened_10m[:h_s2, :w_s2]
    lst_sans_residus_10m = lst_sans_residus_10m[:h_s2, :w_s2]
    masque_nan = np.isnan(ndvi_s2_10m)
    lst_sharpened_10m[masque_nan] = np.nan
    lst_sans_residus_10m[masque_nan] = np.nan

    # --- Calcul du RMSE de conservation d'énergie ---
    from sklearn.metrics import mean_squared_error
    
    # RMSE SANS résidus
    lst_sans_res_agg = aggregate_block(np.nan_to_num(lst_sans_residus_10m, nan=np.nanmean(lst_sans_residus_10m)), BLOCK_SIZE_ECO)
    h_agg, w_agg = lst_sans_res_agg.shape
    h_min_v = min(h_agg, lst_70m_2d.shape[0])
    w_min_v = min(w_agg, lst_70m_2d.shape[1])
    y_true_v = lst_70m_2d[:h_min_v, :w_min_v].flatten()
    y_pred_sans = lst_sans_res_agg[:h_min_v, :w_min_v].flatten()
    masque_v1 = np.isfinite(y_true_v) & np.isfinite(y_pred_sans)
    rmse_sans_residus = np.sqrt(mean_squared_error(y_true_v[masque_v1], y_pred_sans[masque_v1])) if np.sum(masque_v1) > 0 else np.nan

    # RMSE AVEC résidus
    lst_avec_res_agg = aggregate_block(np.nan_to_num(lst_sharpened_10m, nan=np.nanmean(lst_sharpened_10m)), BLOCK_SIZE_ECO)
    y_pred_avec = lst_avec_res_agg[:h_min_v, :w_min_v].flatten()
    masque_v2 = np.isfinite(y_true_v) & np.isfinite(y_pred_avec)
    rmse_avec_residus = np.sqrt(mean_squared_error(y_true_v[masque_v2], y_pred_avec[masque_v2])) if np.sum(masque_v2) > 0 else np.nan

    LOGGER.info(f"   ⚖️  RMSE Conservation d'Énergie : Sans résidus={rmse_sans_residus:.3f}°C | Avec résidus={rmse_avec_residus:.3f}°C")

    # 6. Sauvegarde TIF à 10m
    ds_base = rioxarray.open_rasterio(fichier_ndvi_s2)
    ds_out = ds_base.copy()
    ds_out.values = [lst_sharpened_10m]
    ds_out.rio.to_raster(fichier_sortie)
    LOGGER.info(f"   TIF HD 10m TsHARP Fusion sauvegardé : {fichier_sortie}")

    # 7. Sauvegarde visuelle PNG - Comparaison 3 panneaux
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    vmin = np.nanpercentile(lst_eco_2d, 2)
    vmax = np.nanpercentile(lst_eco_2d, 98)

    # Panneau 1 : Thermique ECOSTRESS original
    im0 = axes[0].imshow(lst_eco_2d, cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title("ECOSTRESS LST ~70m (original)", fontsize=13)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axis('off')

    # Panneau 2 : TsHARP Fusion SANS résidus
    im1 = axes[1].imshow(lst_sans_residus_10m, cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"TsHARP Fusion sans résidus (RMSE={rmse_sans_residus:.2f}°C)", fontsize=13)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')

    # Panneau 3 : TsHARP Fusion AVEC résidus
    im2 = axes[2].imshow(lst_sharpened_10m, cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"TsHARP Fusion avec résidus (RMSE={rmse_avec_residus:.2f}°C)", fontsize=13)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    fig.suptitle(f"{nom_site} – {eco_date_str} (ECOSTRESS+S2 delta={delta_minutes:.0f}min) | Comparaison TsHARP Fusion", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fichier_comparaison, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    LOGGER.info("========================================")
    LOGGER.info("DÉMARRAGE DU TsHARP FUSION (ECOSTRESS Thermique + S2 NDVI)")
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
            parts = nom_fichier.split('_')
            eco_date_str = f"{parts[0]}_{parts[1]}"
            
            s2_date_str, delta_minutes = find_s2_match(nom_site, eco_date_str, TIME_MARGIN_MINUTES)
            
            if s2_date_str:
                process_tsharp_fusion(
                    nom_site, eco_date_str, s2_date_str, delta_minutes,
                    dossier_ecostress, dossier_s2
                )
                nb_fusions += 1
            else:
                LOGGER.info(f"   {eco_date_str} : Pas de paire S2 (<{TIME_MARGIN_MINUTES} min).")
        
        LOGGER.info(f"   {nb_fusions} fusion(s) réalisée(s) pour {nom_site}.")

    LOGGER.info("\nTraitement TsHARP Fusion (ECOSTRESS+S2) terminé pour tous les sites !")

if __name__ == "__main__":
    main()
