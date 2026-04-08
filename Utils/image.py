import numpy as np
from scipy.ndimage import median_filter as scipy_median

from config import LOGGER

def landsat_dn_to_reflectance(dn_array):
    """
    Convertit les Digital Numbers (DN) de Landsat Collection 2 Level 2 
    en valeurs de réflectance de surface (0.0 - 1.0).
    """
    reflectance = (dn_array * 0.0000275) - 0.2
    return reflectance

def get_landsat_mask(qa_values):
    qa_int = qa_values.astype(int)
    cloud = (qa_int >> 3) & 1
    shadow = (qa_int >> 4) & 1
    dilated = (qa_int >> 1) & 1
    cirrus = (qa_int >> 2) & 1
    
    mask = ((cloud == 0) & (shadow == 0) & (dilated == 0) & (cirrus == 0)).astype(np.uint8)
    return mask

def median_filter_2d(dn_array, kernel_size=3):
    filtered = np.empty_like(dn_array)
    for i in range(dn_array.shape[0]):
        band = dn_array[i].copy()
        mask = np.isnan(band)
        
        # Astuce : On remplace temporairement les NaN par la médiane de la bande
        if np.any(~mask):
            fill_value = np.nanmedian(band)
            band_filled = np.nan_to_num(band, nan=fill_value)
            
            f_band = scipy_median(band_filled, size=kernel_size)
            f_band[mask] = np.nan # On remet les trous d'origine
            filtered[i] = f_band
        else:
            filtered[i] = np.nan
            
    return filtered

def stretch_z_score(dn_array, z_min=-2.0, z_max=2.0):
    stretched = np.empty_like(dn_array)
    for i in range(dn_array.shape[0]):
        band = dn_array[i]
        mean, std = np.nanmean(band), np.nanstd(band)
        
        if std == 0 or np.isnan(std):
            stretched[i] = band
            continue
            
        z_score = (band - mean) / std
        # On clip et on normalise entre 0 et 1
        z_clipped = np.clip(z_score, z_min, z_max)
        stretched[i] = (z_clipped - z_min) / (z_max - z_min)
        
    return stretched

def stretch_iqr(dn_array):
    # Calcul des quartiles globaux pour l'image
    q1 = np.nanpercentile(dn_array, 25)
    q3 = np.nanpercentile(dn_array, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Sécurité division par zéro
    if np.isnan(lower_bound) or np.isnan(upper_bound) or (upper_bound == lower_bound):
        return dn_array
    
    stretched = np.clip(dn_array, lower_bound, upper_bound)
    return (stretched - lower_bound) / (upper_bound - lower_bound)

def filtre_median_inteligent(masked_cube):
    LOGGER.info("      🎯 Application du filtre intelligent (Gap-Filling)...")   
    # 1. On isole l'image d'ancrage (qui est toujours en position 0 dans notre liste fusionnée)
    image_principale = masked_cube.isel(time=0)
    # 2. On calcule la "roue de secours" (la médiane de toutes les images disponibles)
    image_secours_mediane = masked_cube.median(dim="time", skipna=True)
    # 3. La magie Xarray : Prends l'image principale, et bouche SES trous avec la médiane !
    image_reparee = image_principale.fillna(image_secours_mediane)
    
    return image_reparee


def calcul_couverture(ds_time):
    total_pixels = ds_time.red.size
    pixels_valides = np.count_nonzero(ds_time.red.values > 0)
    return (pixels_valides / total_pixels) * 100