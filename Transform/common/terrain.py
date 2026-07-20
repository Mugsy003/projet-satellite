"""
Transform/common/terrain.py
============================
Calcul de la Pente (Slope) et de l'Orientation (Aspect) à partir du DEM.
Utilise les gradients numpy pour dériver ces variables morphologiques
à partir du Modèle Numérique de Terrain (MNT) Copernicus GLO-30.

Output :
  - Slope en degrés (0° = plat, 90° = vertical)
  - Aspect décomposé en sin(aspect) et cos(aspect) pour éviter la
    discontinuité circulaire 0°/360°.
"""

import os
import numpy as np
import rioxarray

from config import LOGGER, SITES_PILOTES

DOSSIER_BASE = r"Outputs"


def compute_slope_aspect(dem_array, resolution=30.0):
    """
    Calcule la pente (degrés) et l'orientation (radians) depuis un DEM 2D.

    Parameters
    ----------
    dem_array : np.ndarray (2D)
        Matrice d'altitude en mètres.
    resolution : float
        Taille du pixel en mètres (par défaut 30m pour Copernicus GLO-30).

    Returns
    -------
    slope_deg : np.ndarray (2D)
        Pente en degrés.
    sin_aspect : np.ndarray (2D)
        Sinus de l'orientation (composante E-W).
    cos_aspect : np.ndarray (2D)
        Cosinus de l'orientation (composante N-S).
    """
    # np.gradient retourne (dz/dy, dz/dx) en unités de pixels
    # On divise par la résolution pour avoir la pente en m/m
    dz_dy, dz_dx = np.gradient(dem_array, resolution)

    # Pente = arctan(sqrt(dz_dx² + dz_dy²))
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    # Aspect = arctan2(-dz_dx, dz_dy)  →  0° = Nord, 90° = Est
    # Convention : axe Y positif = vers le Sud (lignes croissantes)
    # donc dz_dy positif = pente descendante vers le Sud
    aspect_rad = np.arctan2(-dz_dx, dz_dy)

    # Décomposition circulaire pour éviter la discontinuité 0°/360°
    sin_aspect = np.sin(aspect_rad)
    cos_aspect = np.cos(aspect_rad)

    return slope_deg, sin_aspect, cos_aspect


def generate_terrain_features(site_name, save=True):
    """
    Charge le DEM d'un site et calcule les features terrain (Slope, sin/cos Aspect).

    Parameters
    ----------
    site_name : str
        Nom du site (clé de SITES_PILOTES).
    save : bool
        Si True, sauvegarde les résultats en TIF alignés sur le DEM.

    Returns
    -------
    dict avec les clés 'slope', 'sin_aspect', 'cos_aspect' (np.ndarray 2D)
    et 'profile' (métadonnées raster : CRS, transform, shape).
    """
    dossier_tif = os.path.join(
        DOSSIER_BASE,
        f"Serie_Temporelle_{site_name}",
        "3_Indices",
        "TIF_Data",
    )
    dem_path = os.path.join(dossier_tif, f"{site_name}_MNT.tif")

    if not os.path.exists(dem_path):
        LOGGER.warning(f"   ⚠️ DEM introuvable pour {site_name} : {dem_path}")
        return None

    LOGGER.info(f"   ⛰️ Chargement du DEM pour {site_name}...")
    ds_dem = rioxarray.open_rasterio(dem_path).squeeze()
    dem_array = ds_dem.values.astype(np.float32)

    # Remplacer les NaN par la médiane pour éviter les artefacts de gradient
    mask_nan = np.isnan(dem_array)
    if mask_nan.any():
        dem_array[mask_nan] = np.nanmedian(dem_array)

    # Résolution du pixel (en mètres) depuis le transform
    res_x = abs(ds_dem.rio.resolution()[0])
    res_y = abs(ds_dem.rio.resolution()[1])
    resolution = (res_x + res_y) / 2.0

    LOGGER.info(f"   📐 Résolution DEM : {resolution:.1f}m")
    LOGGER.info(f"   🔧 Calcul Slope & Aspect...")

    slope_deg, sin_aspect, cos_aspect = compute_slope_aspect(dem_array, resolution)

    profile = {
        "crs": ds_dem.rio.crs,
        "transform": ds_dem.rio.transform(),
        "shape": dem_array.shape,
    }

    if save:
        # Sauvegarder les TIF en utilisant le même template que le DEM
        for name, data in [
            ("Slope", slope_deg),
            ("SinAspect", sin_aspect),
            ("CosAspect", cos_aspect),
        ]:
            out_path = os.path.join(dossier_tif, f"{site_name}_{name}.tif")
            out_da = ds_dem.copy(data=data)
            out_da.rio.to_raster(out_path)
            LOGGER.info(f"   ✅ {name} sauvegardé : {out_path}")

    return {
        "slope": slope_deg,
        "sin_aspect": sin_aspect,
        "cos_aspect": cos_aspect,
        "profile": profile,
    }


if __name__ == "__main__":
    LOGGER.info("=" * 60)
    LOGGER.info("⛰️ CALCUL DES FEATURES TERRAIN (Slope, Aspect)")
    LOGGER.info("=" * 60)

    for site in SITES_PILOTES.keys():
        LOGGER.info(f"\n🌍 Site : {site}")
        result = generate_terrain_features(site, save=True)
        if result:
            LOGGER.info(
                f"   📊 Slope : [{result['slope'].min():.1f}°, {result['slope'].max():.1f}°]"
            )

    LOGGER.info("\n✅ CALCUL TERRAIN TERMINÉ.")
