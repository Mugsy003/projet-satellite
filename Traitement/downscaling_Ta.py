"""
Traitement/downscaling_Ta.py
=============================
Stepwise Downscaling de la température de l'air (Ta) ERA5-Land
de 10 500 m à 1 050 m via une cascade de Random Forest avec
correction résiduelle à chaque étape.

Algorithme (inspiré de la littérature sur le downscaling statistique) :
  1. Charger les prédicteurs à 30m (DEM, Slope, Aspect, NDVI, NDWI, MNDWI, NDBI)
  2. Charger le champ 2D de Ta ERA5-Land (~10.5 km)
  3. Boucle sur les paliers : [10500, 8400, 4200, 2100, 1050] m
     a) Agréger les prédicteurs à R_coarse et R_fine
     b) Rééchantillonner Ta à R_coarse
     c) Entraîner RF(X=prédicteurs@R_coarse, Y=Ta@R_coarse)
     d) Prédire Ta à R_fine
     e) Corriger avec le résidu interpolé bilinéairement
  4. Sauvegarder le raster final Ta@1050m

Usage :
    python Traitement/downscaling_Ta.py --site Selhausen --date 2023-06-16
    python Traitement/downscaling_Ta.py --all
"""

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestRegressor
from pyproj import Transformer

sys.stdout.reconfigure(encoding='utf-8')

from config import LOGGER, SITES_PILOTES, OUTPUT_DIR


# =============================================================================
# CONSTANTES
# =============================================================================

# Cascade de résolutions (en mètres)
RESOLUTIONS = [10500, 8400, 4200, 2100, 1050]

# Paramètres du Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 12,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'n_jobs': -1,
    'random_state': 42,
}

# Résolution native des prédicteurs
RES_NATIVE = 30  # mètres

DOSSIER_SORTIE = "Outputs_Downscaling"


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def aggregate_to_resolution(array_2d, native_res, target_res):
    """
    Agrège une matrice 2D par blocs pour passer d'une résolution native
    à une résolution cible plus grossière.

    Parameters
    ----------
    array_2d : np.ndarray (2D)
    native_res : float
        Résolution d'entrée en mètres.
    target_res : float
        Résolution cible en mètres (doit être un multiple de native_res).

    Returns
    -------
    np.ndarray (2D) agrégé.
    """
    block_size = int(round(target_res / native_res))
    if block_size <= 1:
        return array_2d.copy()

    h, w = array_2d.shape
    h_new = (h // block_size) * block_size
    w_new = (w // block_size) * block_size

    # Si l'image est plus petite que le block_size, retourner la moyenne globale
    if h_new == 0 or w_new == 0:
        return np.array([[np.nanmean(array_2d)]])

    cropped = array_2d[:h_new, :w_new]

    # Reshape en blocs et moyenne (ignoring NaN)
    blocks = cropped.reshape(h_new // block_size, block_size,
                             w_new // block_size, block_size)
    return np.nanmean(blocks, axis=(1, 3))


def resample_bilinear(array_2d, target_shape):
    """
    Rééchantillonne une matrice 2D vers une forme cible via interpolation
    bilinéaire (scipy.ndimage.zoom).

    Parameters
    ----------
    array_2d : np.ndarray (2D)
    target_shape : tuple (h, w)

    Returns
    -------
    np.ndarray (2D) rééchantillonné.
    """
    if array_2d.shape == target_shape:
        return array_2d.copy()

    zoom_factors = (target_shape[0] / array_2d.shape[0],
                    target_shape[1] / array_2d.shape[1])

    # Remplacer les NaN par la médiane avant interpolation
    mask_nan = np.isnan(array_2d)
    arr_filled = array_2d.copy()
    if mask_nan.any():
        arr_filled[mask_nan] = np.nanmedian(array_2d)

    result = zoom(arr_filled, zoom_factors, order=1)  # order=1 = bilinéaire
    return result


def reproject_era5_to_utm(ta_era5_da, crs_target, transform_target, shape_target):
    """
    Reprojette le champ 2D Ta ERA5 (lat/lon) vers le CRS UTM des rasters Landsat,
    et le rééchantillonne à la résolution cible.

    Parameters
    ----------
    ta_era5_da : xr.DataArray
        Champ 2D de Ta en °C (coordonnées lat/lon).
    crs_target : rasterio.crs.CRS
        CRS cible (UTM du site).
    transform_target : rasterio.Affine
        Transform affine de la grille cible.
    shape_target : tuple (h, w)
        Dimensions de la grille cible.

    Returns
    -------
    np.ndarray (2D) : Ta reprojetée sur la grille UTM cible.
    """
    from rasterio.crs import CRS

    src_data = ta_era5_da.values.astype(np.float32)
    if src_data.ndim == 2:
        src_data = src_data[np.newaxis, :, :]  # (1, H, W)

    # Construire le transform source depuis les coordonnées lat/lon
    lons = ta_era5_da.longitude.values if 'longitude' in ta_era5_da.dims else ta_era5_da.x.values
    lats = ta_era5_da.latitude.values if 'latitude' in ta_era5_da.dims else ta_era5_da.y.values

    # ERA5 : lats décroissantes, lons croissantes
    if lats[0] < lats[-1]:
        lats = lats[::-1]
        src_data = src_data[:, ::-1, :]

    res_lon = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.1
    res_lat = abs(lats[0] - lats[1]) if len(lats) > 1 else 0.1

    src_transform = rasterio.transform.from_bounds(
        lons.min() - res_lon / 2,
        lats.min() - res_lat / 2,
        lons.max() + res_lon / 2,
        lats.max() + res_lat / 2,
        len(lons),
        len(lats),
    )

    NODATA = -9999.0
    dst_data = np.full((1,) + shape_target, NODATA, dtype=np.float32)

    reproject(
        source=src_data,
        destination=dst_data,
        src_transform=src_transform,
        src_crs=CRS.from_epsg(4326),
        dst_transform=transform_target,
        dst_crs=crs_target,
        resampling=Resampling.bilinear,
        src_nodata=NODATA,
        dst_nodata=NODATA,
    )

    result = dst_data.squeeze()
    # Pixels hors emprise ERA5 → NaN
    result[result == NODATA] = np.nan
    # Sécurité : les 0.0 exactement au bord sont aussi suspects
    result[result == 0.0] = np.nan

    return result


# =============================================================================
# CHARGEMENT DES PRÉDICTEURS
# =============================================================================

def load_predictors(site, date_str=None):
    """
    Charge tous les prédicteurs sur une emprise élargie (~60 km) pour
    couvrir le domaine ERA5 du downscaling.

    Le DEM est téléchargé sur l'emprise élargie depuis Copernicus GLO-30
    à la résolution finale cible (1050m).
    Les indices spectraux Landsat (emprise ~6 km) sont reprojetés sur cette
    grille, et étendus par leur valeur médiane en dehors de l'emprise Landsat.
    """
    import odc.stac
    import pystac_client
    import planetary_computer

    coords = SITES_PILOTES[site]
    lon, lat = coords['lon'], coords['lat']

    # Emprise élargie : ±0.3° autour du site (même que l'extraction ERA5 2D)
    margin = 0.30
    bbox_wide = [lon - margin, lat - margin, lon + margin, lat + margin]

    # Dossier de cache pour le DEM élargi
    cache_dir = os.path.join(DOSSIER_SORTIE, "_cache_dem")
    os.makedirs(cache_dir, exist_ok=True)
    dem_wide_path = os.path.join(cache_dir, f"{site}_DEM_wide_1050m.tif")

    # Résolution de travail : 1050m (résolution finale du downscaling)
    WORK_RES = 1050

    if os.path.exists(dem_wide_path):
        LOGGER.info(f"   ✅ DEM élargi déjà en cache : {dem_wide_path}")
        ds_dem = rioxarray.open_rasterio(dem_wide_path).squeeze()
    else:
        LOGGER.info(f"   📥 Téléchargement du DEM élargi (±0.3°, {WORK_RES}m)...")

        # Déterminer le CRS UTM du site depuis les données Landsat existantes
        dossier_tif_landsat = os.path.join(
            "Outputs", f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data"
        )
        dem_landsat_path = os.path.join(dossier_tif_landsat, f"{site}_MNT.tif")
        if os.path.exists(dem_landsat_path):
            ds_template = rioxarray.open_rasterio(dem_landsat_path).squeeze()
            crs_utm = ds_template.rio.crs
        else:
            # Fallback : déterminer l'UTM zone depuis lon/lat
            utm_zone = int((lon + 180) / 6) + 1
            epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
            from rasterio.crs import CRS
            crs_utm = CRS.from_epsg(epsg)

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace
        )
        search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox_wide)

        import time as time_module
        items = []
        for attempt in range(3):
            try:
                items = list(search.items())
                break
            except Exception as e:
                LOGGER.warning(f"      ⚠️ Tentative {attempt+1}/3 : {e}")
                time_module.sleep(3)

        if not items:
            LOGGER.error("   ❌ Impossible de télécharger le DEM élargi.")
            return None, None

        ds_dem_brut = odc.stac.stac_load(
            items, bbox=bbox_wide, bands=["data"],
            crs=crs_utm, resolution=WORK_RES,
            patch_url=planetary_computer.sign
        ).isel(time=0)["data"]

        ds_dem = ds_dem_brut.astype('float32')
        ds_dem = ds_dem.where(ds_dem != ds_dem.rio.nodata, np.nan)
        ds_dem.rio.to_raster(dem_wide_path)
        LOGGER.info(f"   ✅ DEM élargi sauvegardé : {ds_dem.shape}")

    dem = ds_dem.values.astype(np.float32)
    mask_nan = np.isnan(dem)
    if mask_nan.any():
        dem[mask_nan] = np.nanmedian(dem)

    profile = {
        "crs": ds_dem.rio.crs,
        "transform": ds_dem.rio.transform(),
        "shape": dem.shape,
    }

    # Calculer Slope et Aspect depuis le DEM élargi
    from Transform.common.terrain import compute_slope_aspect
    res_x = abs(ds_dem.rio.resolution()[0])
    slope_deg, sin_aspect, cos_aspect = compute_slope_aspect(dem, res_x)

    predictors = {
        'DEM': dem,
        'Slope': slope_deg,
        'SinAspect': sin_aspect,
        'CosAspect': cos_aspect,
    }

    # --- Charger et reprojeter les indices spectraux Landsat ---
    dossier_tif = os.path.join(
        "Outputs", f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data"
    )
    fichiers = os.listdir(dossier_tif) if os.path.exists(dossier_tif) else []

    for idx_name in ['NDVI', 'NDWI', 'NDBI']:
        pattern = f"_{idx_name}.tif"
        if date_str:
            matching = [f for f in fichiers if date_str in f and pattern in f]
        else:
            matching = [f for f in fichiers if pattern in f]

        if matching:
            idx_ds = rioxarray.open_rasterio(
                os.path.join(dossier_tif, matching[0])
            ).squeeze()
            # Reprojeter l'indice Landsat (30m, emprise ~6km) sur la grille élargie (1050m, ~60km)
            idx_reproj = idx_ds.rio.reproject_match(ds_dem).values.astype(np.float32)
            # Hors emprise Landsat → remplir par la médiane des valeurs valides
            valid_vals = idx_reproj[np.isfinite(idx_reproj)]
            fill_val = float(np.median(valid_vals)) if len(valid_vals) > 0 else 0.0
            idx_reproj = np.nan_to_num(idx_reproj, nan=fill_val)
            predictors[idx_name] = idx_reproj
            LOGGER.info(f"      ✅ {idx_name} reprojeté sur grille élargie")
        else:
            LOGGER.warning(f"      ⚠️ {idx_name} non trouvé, rempli par 0")
            predictors[idx_name] = np.zeros_like(dem)

    # MNDWI : pas encore disponible dans le pipeline, rempli par 0
    predictors['MNDWI'] = np.zeros_like(dem)

    # Nettoyage final
    for key in predictors:
        m = np.isnan(predictors[key])
        if m.any():
            predictors[key][m] = float(np.nanmedian(predictors[key]))

    LOGGER.info(f"   📊 {len(predictors)} prédicteurs sur grille élargie "
                f"({dem.shape[0]}x{dem.shape[1]} pixels à ~{WORK_RES}m)")

    return predictors, profile


# =============================================================================
# BOUCLE DE STEPWISE DOWNSCALING
# =============================================================================

def run_stepwise_downscaling(ta_coarse, predictors_native, profile,
                             resolutions=None, native_res=RES_NATIVE):
    """
    Exécute le downscaling stepwise de Ta depuis la résolution la plus
    grossière jusqu'à la plus fine.

    Parameters
    ----------
    ta_coarse : np.ndarray (2D)
        Champ 2D de Ta (°C) à la résolution la plus grossière,
        déjà reprojeté sur la grille UTM des prédicteurs.
    predictors_native : dict
        Prédicteurs à la résolution native (30m).
    profile : dict
        Métadonnées raster (CRS, transform, shape).
    resolutions : list of int
        Liste des paliers de résolution en mètres.
    native_res : float
        Résolution native des prédicteurs.

    Returns
    -------
    np.ndarray (2D) : Champ Ta à la résolution finale (1050m),
                       sur la grille des prédicteurs agrégés.
    """
    if resolutions is None:
        resolutions = RESOLUTIONS

    LOGGER.info(f"\n   🔄 STEPWISE DOWNSCALING : {resolutions[0]}m → {resolutions[-1]}m")
    LOGGER.info(f"      Paliers : {' → '.join(str(r) + 'm' for r in resolutions)}")

    # Agréger Ta initiale à la première résolution de la cascade
    # Ta_coarse est déjà sur la grille UTM mais à une résolution ERA5 (~10.5 km)
    # On la rééchantillonne pour qu'elle ait la forme attendue à R_coarse=10500m
    r0 = resolutions[0]
    shape_r0 = (predictors_native['DEM'].shape[0] * native_res // r0,
                predictors_native['DEM'].shape[1] * native_res // r0)
    shape_r0 = (max(int(shape_r0[0]), 2), max(int(shape_r0[1]), 2))

    ta_current = resample_bilinear(ta_coarse, shape_r0)
    LOGGER.info(f"      Ta initiale rééchantillonnée à {r0}m : {ta_current.shape}")

    for step_i in range(len(resolutions) - 1):
        r_coarse = resolutions[step_i]
        r_fine = resolutions[step_i + 1]

        LOGGER.info(f"\n   --- Étape {step_i + 1}/{len(resolutions) - 1} : "
                    f"{r_coarse}m → {r_fine}m ---")

        # A) Agréger les prédicteurs aux deux résolutions
        X_coarse = {}
        X_fine = {}
        for name, arr in predictors_native.items():
            X_coarse[name] = aggregate_to_resolution(arr, native_res, r_coarse)
            X_fine[name] = aggregate_to_resolution(arr, native_res, r_fine)

        # Vérifier que Ta_current a la bonne forme pour R_coarse
        shape_coarse = X_coarse['DEM'].shape
        if ta_current.shape != shape_coarse:
            ta_current = resample_bilinear(ta_current, shape_coarse)

        # B) Construire les matrices d'entraînement
        feature_names = sorted(X_coarse.keys())
        X_train = np.column_stack([X_coarse[f].ravel() for f in feature_names])
        Y_train = ta_current.ravel()

        # Masquer les pixels invalides
        valid_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(Y_train)
        n_valid = valid_mask.sum()

        if n_valid < 10:
            LOGGER.warning(f"      ⚠️ Seulement {n_valid} pixels valides, "
                          f"interpolation bilinéaire directe.")
            shape_fine = X_fine['DEM'].shape
            ta_current = resample_bilinear(ta_current, shape_fine)
            continue

        LOGGER.info(f"      📊 Entraînement RF : {n_valid} pixels, "
                    f"{len(feature_names)} features")

        # C) Entraîner le Random Forest
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_train[valid_mask], Y_train[valid_mask])

        # Importance des features (diagnostic)
        importances = dict(zip(feature_names, rf.feature_importances_))
        top3 = sorted(importances.items(), key=lambda x: -x[1])[:3]
        LOGGER.info(f"      🌟 Top-3 features : "
                    + ", ".join(f"{k}={v:.2f}" for k, v in top3))

        # D) Prédire à la résolution fine
        shape_fine = X_fine['DEM'].shape
        X_pred = np.column_stack([X_fine[f].ravel() for f in feature_names])
        valid_pred = np.isfinite(X_pred).all(axis=1)

        ta_pred_fine = np.full(X_pred.shape[0], np.nan)
        if valid_pred.any():
            ta_pred_fine[valid_pred] = rf.predict(X_pred[valid_pred])
        ta_pred_fine = ta_pred_fine.reshape(shape_fine)

        # E) Correction résiduelle
        # Prédire à R_coarse pour calculer le résidu
        ta_pred_coarse = np.full(X_train.shape[0], np.nan)
        ta_pred_coarse[valid_mask] = rf.predict(X_train[valid_mask])
        ta_pred_coarse = ta_pred_coarse.reshape(shape_coarse)

        residual_coarse = ta_current - ta_pred_coarse
        residual_coarse = np.nan_to_num(residual_coarse, nan=0.0)

        # Interpoler le résidu à R_fine (bilinéaire)
        residual_fine = resample_bilinear(residual_coarse, shape_fine)

        # Ta corrigée = prédiction + résidu interpolé
        ta_corrected = ta_pred_fine + residual_fine

        # Remplacer les NaN par la valeur interpolée depuis la résolution précédente
        nan_mask = np.isnan(ta_corrected)
        if nan_mask.any():
            ta_fallback = resample_bilinear(ta_current, shape_fine)
            ta_corrected[nan_mask] = ta_fallback[nan_mask]

        # Statistiques de l'étape
        rmse_residual = np.sqrt(np.nanmean(residual_coarse**2))
        LOGGER.info(f"      ✅ Ta@{r_fine}m : [{np.nanmin(ta_corrected):.1f}°C, "
                    f"{np.nanmax(ta_corrected):.1f}°C], "
                    f"RMSE résidu = {rmse_residual:.3f}°C")

        # F) La sortie de cette étape devient l'entrée de la suivante
        ta_current = ta_corrected

    return ta_current


# =============================================================================
# SAUVEGARDE DU RASTER FINAL
# =============================================================================

def save_ta_raster(ta_array, profile_native, target_res, site, date_str, output_dir=None):
    """
    Sauvegarde le raster de Ta downscalé en GeoTIFF.

    Parameters
    ----------
    ta_array : np.ndarray (2D)
        Champ Ta à la résolution finale.
    profile_native : dict
        Profil raster natif (CRS, transform à 30m).
    target_res : int
        Résolution finale en mètres.
    site : str
    date_str : str
    output_dir : str, optional

    Returns
    -------
    str : Chemin du fichier sauvegardé.
    """
    if output_dir is None:
        output_dir = DOSSIER_SORTIE
    os.makedirs(output_dir, exist_ok=True)

    # Recalculer le transform pour la résolution cible
    native_transform = profile_native['transform']
    native_shape = profile_native['shape']

    # Emprise (bounds) de la grille native
    left = native_transform.c
    top = native_transform.f
    right = left + native_shape[1] * abs(native_transform.a)
    bottom = top - native_shape[0] * abs(native_transform.e)

    # Nouveau transform pour la résolution cible
    new_transform = from_bounds(left, bottom, right, top,
                                ta_array.shape[1], ta_array.shape[0])

    output_path = os.path.join(output_dir, f"Ta_downscaled_{site}_{date_str}_{target_res}m.tif")

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=ta_array.shape[0],
        width=ta_array.shape[1],
        count=1,
        dtype='float32',
        crs=profile_native['crs'],
        transform=new_transform,
        nodata=np.nan,
    ) as dst:
        dst.write(ta_array.astype(np.float32), 1)

    LOGGER.info(f"   💾 Raster Ta sauvegardé : {output_path}")
    LOGGER.info(f"      Résolution : {target_res}m, Shape : {ta_array.shape}")
    return output_path


# =============================================================================
# ORCHESTRATEUR POUR UN SITE / UNE DATE
# =============================================================================

def process_site_date(site, date_str):
    """
    Exécute le downscaling complet pour un site et une date donnés.

    Parameters
    ----------
    site : str
    date_str : str (format YYYY-MM-DD)

    Returns
    -------
    str : Chemin du raster final, ou None en cas d'erreur.
    """
    LOGGER.info(f"\n{'='*60}")
    LOGGER.info(f"🌡️ DOWNSCALING Ta : {site} — {date_str}")
    LOGGER.info(f"{'='*60}")

    coords = SITES_PILOTES[site]

    # 1. Charger les prédicteurs sur la grille élargie (1050m)
    LOGGER.info("\n   📦 Chargement des prédicteurs (grille élargie ~60km)...")
    predictors, profile = load_predictors(site, date_str)
    if predictors is None:
        return None

    # 2. Charger / télécharger le champ 2D de Ta ERA5
    LOGGER.info("\n   🌐 Chargement du champ 2D de Ta ERA5-Land...")
    from Extraction.ERA5.extraction_ERA5 import download_era5_ta_2d

    target_dt = pd.to_datetime(f"{date_str} 10:30:00")
    ta_era5 = download_era5_ta_2d(site, coords['lon'], coords['lat'], target_dt)

    if ta_era5 is None:
        LOGGER.error("   ❌ Impossible de charger le champ 2D de Ta ERA5")
        return None

    # 3. Reprojeter Ta ERA5 (lat/lon) vers la grille UTM des prédicteurs
    LOGGER.info("   📐 Reprojection de Ta vers le CRS UTM du site...")
    ta_utm = reproject_era5_to_utm(
        ta_era5,
        crs_target=profile['crs'],
        transform_target=profile['transform'],
        shape_target=profile['shape'],
    )
    LOGGER.info(f"      Ta reprojetée : {ta_utm.shape}, "
                f"[{np.nanmin(ta_utm):.1f}°C, {np.nanmax(ta_utm):.1f}°C]")

    # 4. Stepwise Downscaling (prédicteurs à 1050m)
    ta_final = run_stepwise_downscaling(ta_utm, predictors, profile, native_res=1050)

    # 5. Sauvegarde
    output_path = save_ta_raster(
        ta_final, profile, RESOLUTIONS[-1], site, date_str
    )

    return output_path


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stepwise Downscaling de Ta ERA5-Land (10.5km → 1km)"
    )
    parser.add_argument('--site', type=str, help='Nom du site (ex: Selhausen)')
    parser.add_argument('--date', type=str, help='Date au format YYYY-MM-DD')
    parser.add_argument('--all', action='store_true',
                        help='Traiter tous les sites et dates disponibles')
    args = parser.parse_args()

    LOGGER.info("=" * 60)
    LOGGER.info("🌡️ STEPWISE DOWNSCALING — Ta ERA5-Land (10.5km → 1km)")
    LOGGER.info("=" * 60)

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)

    if args.all:
        # Traiter tous les sites pour toutes les dates disponibles
        for site in SITES_PILOTES.keys():
            dossier_tif = os.path.join(
                "Outputs", f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data"
            )
            if not os.path.exists(dossier_tif):
                LOGGER.warning(f"⚠️ Pas de données TIF pour {site}, skip.")
                continue

            # Lister les dates disponibles (depuis les fichiers NDVI)
            fichiers = os.listdir(dossier_tif)
            dates = set()
            for f in fichiers:
                match = re.search(r"(\d{4}-\d{2}-\d{2})", f)
                if match and 'NDVI' in f:
                    dates.add(match.group(1))

            for date_str in sorted(dates):
                try:
                    process_site_date(site, date_str)
                except Exception as e:
                    LOGGER.error(f"❌ Erreur {site}/{date_str} : {e}")
                    import traceback
                    LOGGER.error(traceback.format_exc())

    elif args.site and args.date:
        process_site_date(args.site, args.date)

    else:
        parser.print_help()
        print("\nExemples :")
        print("  python Traitement/downscaling_Ta.py --site Selhausen --date 2023-06-16")
        print("  python Traitement/downscaling_Ta.py --all")

    LOGGER.info(f"\n{'='*60}")
    LOGGER.info("✅ DOWNSCALING TERMINÉ.")
    LOGGER.info(f"{'='*60}")


if __name__ == "__main__":
    main()
