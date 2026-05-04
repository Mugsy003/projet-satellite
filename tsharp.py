"""TsHARP (Thermal SHARPening) baseline algorithm.

Implements the classic TsHARP approach for thermal image super-resolution
using the quadratic relationship between NDVI and Land Surface Temperature:
    LST = a * NDVI^2 + b * NDVI + c

References:
    Agam et al. (2007) "A vegetation index based technique for spatial
    sharpening of thermal imagery"
"""

import numpy as np
from scipy.ndimage import zoom


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Compute Normalized Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)

    Args:
        nir: Near-infrared band array.
        red: Red band array.

    Returns:
        NDVI array clipped to [-1, 1].
    """
    numerator = nir.astype(np.float64) - red.astype(np.float64)
    denominator = nir.astype(np.float64) + red.astype(np.float64)
    ndvi = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def fit_tsharp(
    coarse_lst: np.ndarray,
    coarse_ndvi: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Fit quadratic regression LST = a*NDVI^2 + b*NDVI + c.

    Uses ordinary least-squares to find coefficients [a, b, c] that best
    describe the LST-NDVI relationship at coarse resolution.

    Args:
        coarse_lst: Coarse-resolution LST array (2D).
        coarse_ndvi: Coarse-resolution NDVI array (2D, same shape as coarse_lst).
        mask: Optional boolean mask. True where pixels are valid.

    Returns:
        Coefficient array [a, b, c] of shape (3,).
    """
    lst_flat = coarse_lst.ravel().astype(np.float64)
    ndvi_flat = coarse_ndvi.ravel().astype(np.float64)

    if mask is not None:
        mask_flat = mask.ravel().astype(bool)
        lst_flat = lst_flat[mask_flat]
        ndvi_flat = ndvi_flat[mask_flat]

    # Build design matrix: [NDVI^2, NDVI, 1]
    design = np.column_stack([ndvi_flat**2, ndvi_flat, np.ones_like(ndvi_flat)])

    # Solve least squares: design @ [a, b, c]^T = lst
    coefficients, _, _, _ = np.linalg.lstsq(design, lst_flat, rcond=None)

    return coefficients.astype(np.float64)


def predict_tsharp(
    coarse_lst: np.ndarray,
    coarse_ndvi: np.ndarray,
    fine_ndvi: np.ndarray,
    coefficients: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Predict fine-resolution LST using TsHARP.

    Steps:
        1. Fit (or use provided) quadratic coefficients from coarse data.
        2. Compute coarse residual: residual = coarse_LST - predicted_coarse_LST.
        3. Interpolate residual to fine resolution via scipy.ndimage.zoom.
        4. Predict fine LST: LST_fine = a*fine_NDVI^2 + b*fine_NDVI + c + residual_interp.

    Args:
        coarse_lst: Coarse-resolution LST array (2D).
        coarse_ndvi: Coarse-resolution NDVI array (2D, same shape as coarse_lst).
        fine_ndvi: Fine-resolution NDVI array (2D).
        coefficients: Optional pre-computed coefficients [a, b, c]. If None,
            they are fitted from the coarse data.
        mask: Optional boolean mask for fitting. True where pixels are valid.

    Returns:
        Fine-resolution LST prediction (2D, same shape as fine_ndvi).
    """
    if coefficients is None:
        coefficients = fit_tsharp(coarse_lst, coarse_ndvi, mask=mask)

    a, b, c = coefficients

    # Predicted coarse LST from the regression
    coarse_predicted = (
        a * coarse_ndvi.astype(np.float64) ** 2
        + b * coarse_ndvi.astype(np.float64)
        + c
    )

    # Coarse residual
    residual = coarse_lst.astype(np.float64) - coarse_predicted

    # Interpolate residual to fine resolution
    zoom_factors = (
        fine_ndvi.shape[0] / coarse_lst.shape[0],
        fine_ndvi.shape[1] / coarse_lst.shape[1],
    )
    residual_interp = zoom(residual, zoom_factors, order=1)

    # Predict fine-resolution LST
    fine_predicted = (
        a * fine_ndvi.astype(np.float64) ** 2
        + b * fine_ndvi.astype(np.float64)
        + c
        + residual_interp
    )

    return fine_predicted.astype(np.float32)

import os
import glob
import rioxarray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import SITES_PILOTES, LOGGER

DOSSIER_BASE = r"Outputs"

def aggregate_3x3(matrice_2d):
    """Regroupe les pixels par blocs de 3x3 et calcule la moyenne.
    Permet de repasser d'une grille de 30m à une grille de ~90m."""
    h, w = matrice_2d.shape
    h_new = (h // 3) * 3
    w_new = (w // 3) * 3
    matrice_coupee = matrice_2d[:h_new, :w_new]
    return matrice_coupee.reshape(h_new // 3, 3, w_new // 3, 3).mean(axis=(1, 3))

def load_raster_as_2d(filepath):
    """Charge un fichier TIF et retourne sa matrice 2D."""
    ds = rioxarray.open_rasterio(filepath)
    profile = {"crs": ds.rio.crs, "transform": ds.rio.transform(), "shape": ds.shape}
    array_2d = ds.values.squeeze()
    return array_2d, profile

def process_tsharp_for_image(nom_site, date_str, dossier_indices):
    """Exécute l'algorithme TsHARP."""
    LOGGER.info(f"\n   📅 Traitement de l'image du {date_str} (Méthode TsHARP)...")

    fichier_thermique = os.path.join(dossier_indices, f"{date_str}_{nom_site}_LST.tif")
    fichier_ndvi = os.path.join(dossier_indices, f"{date_str}_{nom_site}_NDVI.tif")
    fichier_sortie = os.path.join(dossier_indices, f"{date_str}_{nom_site}_LST_Sharpened_TsHARP.tif")
    fichier_comparaison = os.path.join(dossier_indices, f"{date_str}_{nom_site}_Comparaison_TsHARP.png")

    if not os.path.exists(fichier_thermique) or not os.path.exists(fichier_ndvi):
        LOGGER.warning(f"   ❌ Fichiers LST ou NDVI manquants pour {date_str}. Ignoré.")
        return

    lst_30m_2d, raster_profile = load_raster_as_2d(fichier_thermique)
    ndvi_30m_2d, _ = load_raster_as_2d(fichier_ndvi)

    h, w = lst_30m_2d.shape
    h_actual, w_actual = ndvi_30m_2d.shape
    if h_actual < h or w_actual < w:
        LOGGER.warning(f"   ⚠️  NDVI a des dimensions insuffisantes ({h_actual}x{w_actual} vs {h}x{w}). Ignoré.")
        return
    ndvi_30m_2d = ndvi_30m_2d[:h, :w]

    lst_90m_2d = aggregate_3x3(lst_30m_2d)
    ndvi_90m_2d = aggregate_3x3(ndvi_30m_2d)

    mask_90m = np.isfinite(lst_90m_2d) & np.isfinite(ndvi_90m_2d)

    try:
        # Comme "zoom" (dans predict_tsharp) ne gère pas bien les NaNs,
        # on peut remplacer temporairement les NaNs de lst_90m_2d par la moyenne ou 0
        # Mais on laisse predict_tsharp faire son travail pour le moment.
        # Si un warning intervient, il sera intercepté.
        lst_sharpened_30m_2d = predict_tsharp(
            coarse_lst=np.nan_to_num(lst_90m_2d, nan=np.nanmean(lst_90m_2d)),
            coarse_ndvi=np.nan_to_num(ndvi_90m_2d, nan=np.nanmean(ndvi_90m_2d)),
            fine_ndvi=np.nan_to_num(ndvi_30m_2d, nan=np.nanmean(ndvi_30m_2d)),
            mask=mask_90m
        )
    except Exception as e:
        LOGGER.error(f"   ❌ Erreur lors de TsHARP : {e}")
        return

    # Ajuster la taille finale car zoom(..., zoom_factors) peut donner des tailles légèrement différentes
    lst_sharpened_30m_2d = lst_sharpened_30m_2d[:h, :w]

    # Remettre les NaNs là où ils étaient
    masque_nan = np.isnan(lst_30m_2d)
    lst_sharpened_30m_2d[masque_nan] = np.nan

    # Sauvegarde TIF
    ds_base = rioxarray.open_rasterio(fichier_thermique)
    ds_out = ds_base.copy()
    ds_out.values = [lst_sharpened_30m_2d]
    ds_out.rio.to_raster(fichier_sortie)
    LOGGER.info(f"   💾 TIF HD TsHARP sauvegardé : {fichier_sortie}")

    # Sauvegarde visuelle PNG
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(lst_30m_2d, cmap='magma', vmin=10, vmax=50) 
    plt.title("Avant : Thermique 100m (Interpolé NASA)", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lst_sharpened_30m_2d, cmap='magma', vmin=10, vmax=50)
    plt.title("Après : TsHARP", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(fichier_comparaison, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    LOGGER.info("========================================")
    LOGGER.info("🔥 DÉMARRAGE DU DOWNSCALING TsHARP POUR TOUS LES SITES")
    LOGGER.info("========================================")

    for nom_site in SITES_PILOTES.keys():
        LOGGER.info(f"\n🌍 === Traitement du site : {nom_site} ===")
        
        dossier_indices = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}", "3_Indices", "TIF_Data")
        
        if not os.path.exists(dossier_indices):
            continue
            
        fichiers_thermiques = glob.glob(os.path.join(dossier_indices, f"*_{nom_site}_LST.tif"))
        
        for chemin_fichier in fichiers_thermiques:
            nom_fichier = os.path.basename(chemin_fichier)
            parts = nom_fichier.split('_')
            date_str = f"{parts[0]}_{parts[1]}"
            process_tsharp_for_image(nom_site, date_str, dossier_indices)

    LOGGER.info("\n✅ Traitement TsHARP terminé pour tous les sites et toutes les dates !")

if __name__ == "__main__":
    main()
