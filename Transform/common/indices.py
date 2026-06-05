"""
Transform/indices.py
Regroupe toutes les fonctions de calcul d'indices spectraux.
"""
import numpy as np
from config import LOGGER

def safe_divide(numerator, denominator):
    return np.divide(
        numerator, 
        denominator, 
        out=np.full_like(denominator, np.nan, dtype=float), 
        where=(denominator != 0)
    )

def calculate_ndvi(red, nir):
    LOGGER.info("      📊 Calcul de l'indice NDVI (Végétation)...")
    return safe_divide(nir - red, nir + red)

def calculate_ndwi(green, nir):
    LOGGER.info("      💧 Calcul de l'indice NDWI (Eau)...")
    return safe_divide(green - nir, green + nir)

def calculate_ndbi(swir, nir):
    LOGGER.info("      🏙️ Calcul de l'indice NDBI (Sols nus/Bâti)...")
    return safe_divide(swir - nir, swir + nir)

def calculate_evi(red, nir, blue):
    LOGGER.info("      🌿 Calcul de l'indice EVI (Végétation améliorée)...")
    denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    return safe_divide(2.5 * (nir - red), denominator)

def calculate_savi(red, nir, L=0.5):
    LOGGER.info("      🌱 Calcul de l'indice SAVI (Végétation ajustée au sol)...")
    return safe_divide((nir - red) * (1 + L), nir + red + L)

def calculate_lst_step_by_step(bt_kelvin, ndvi_array):
    LOGGER.info("      🌡️ Début du calcul LST (avec Émissivité empirique par paliers)...")
    
    #Estimation directe de l'Émissivité (E)
    
    # 1. On définit nos 4 conditions (masques booléens)
    cond_eau = ndvi_array < -0.185
    cond_sol_nu = (ndvi_array >= -0.185) & (ndvi_array < 0.157)
    cond_mixte = (ndvi_array >= 0.157) & (ndvi_array <= 0.727)
    cond_dense = ndvi_array > 0.727
    
    # Pour éviter les problèmes de log(0) dans la formule du mixte, on remplace les NDVI négatifs ou nuls par une petite valeur positive
    ndvi_safe_for_log = np.where(ndvi_array <= 0, 0.1, ndvi_array)
    
    emissivity = np.select(
        [cond_eau, cond_sol_nu, cond_mixte, cond_dense],
        [
            0.995,                                           # Eau
            0.970,                                           # Sol nu
            1.0094 + 0.047 * np.log(ndvi_safe_for_log),      # Mixte
            0.990                                            # Végétation dense
        ],
        default=0.98  # Sécurité au cas où un pixel échappe à toutes les règles (NaN)
    )
    
    #Calculation of Land Surface Temperature (LST)
    lambda_val = 10.8e-6  # Longueur d'onde effective de la Bande 10
    rho = 1.438e-2        # Constante physique de Planck/Boltzmann
    
    # Formule finale de la LST (résultat en Kelvin)
    lst_kelvin = bt_kelvin / (1 + (lambda_val * bt_kelvin / rho) * np.log(emissivity))
    
    # Conversion en degrés Celsius 
    lst_celsius = lst_kelvin - 273.15
    
    return lst_celsius