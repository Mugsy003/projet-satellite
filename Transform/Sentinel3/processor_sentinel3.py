"""
Transform/Sentinel3/processor_sentinel3.py
==========================================
Processeur pour les données Sentinel-3 (format NetCDF, géométrie Swath).

Ce module télécharge les fichiers NetCDF bruts depuis Planetary Computer,
les reprojette sur une grille régulière géoréférencée, et exporte en TIF
standard compatible avec le reste du pipeline (DMS sharpening, etc.).

Résolutions :
  - SLSTR LST : ~1 km natif → reprojeté sur grille ~1 km
  - Synergy Optique (SDR) : ~300 m natif → reprojeté sur grille ~300 m
"""
import os
import numpy as np
import xarray as xr
import urllib.request
import rasterio
from rasterio.transform import from_bounds
from pyproj import Transformer
from scipy.interpolate import griddata
from config import LOGGER


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def download_netcdf_asset(href, dest_path):
    """Télécharge un fichier NetCDF depuis un lien STAC signé."""
    if not os.path.exists(dest_path):
        LOGGER.info(f"      📥 Téléchargement : {os.path.basename(dest_path)}")
        urllib.request.urlretrieve(href, dest_path)
    return dest_path


def _get_utm_epsg(lon_center, lat_center):
    """Calcule le code EPSG UTM à partir d'une coordonnée centrale."""
    zone_number = int((lon_center + 180) / 6) + 1
    if lat_center >= 0:
        return f"EPSG:{32600 + zone_number}"
    else:
        return f"EPSG:{32700 + zone_number}"


def _reproject_swath_to_grid(data_2d, lon_2d, lat_2d, bbox_lonlat, target_res_m, method='nearest'):
    """
    Reprojette un swath irrégulier (lat/lon par pixel) sur une grille UTM régulière.

    Paramètres :
        data_2d   : np.ndarray 2D (rows, cols) – les valeurs brutes du swath
        lon_2d    : np.ndarray 2D – longitude de chaque pixel
        lat_2d    : np.ndarray 2D – latitude de chaque pixel
        bbox_lonlat : (minlon, minlat, maxlon, maxlat)
        target_res_m : résolution cible en mètres (ex: 1000 pour 1km, 300 pour 300m)
        method    : méthode d'interpolation pour griddata ('nearest', 'linear', 'cubic')

    Retourne :
        grid_data   : np.ndarray 2D reprojeté sur la grille régulière
        transform   : rasterio.Affine
        crs_str     : chaîne EPSG (ex: "EPSG:32631")
        (height, width) : dimensions de la grille
    """
    minlon, minlat, maxlon, maxlat = bbox_lonlat

    # 1. Déterminer la zone UTM à partir du centre de la bbox
    lon_center = (minlon + maxlon) / 2
    lat_center = (minlat + maxlat) / 2
    crs_str = _get_utm_epsg(lon_center, lat_center)

    # 2. Transformer la bbox en coordonnées UTM
    transformer = Transformer.from_crs("EPSG:4326", crs_str, always_xy=True)
    x_min, y_min = transformer.transform(minlon, minlat)
    x_max, y_max = transformer.transform(maxlon, maxlat)

    # 3. Créer la grille cible régulière
    width = int(np.ceil((x_max - x_min) / target_res_m))
    height = int(np.ceil((y_max - y_min) / target_res_m))

    if width < 2 or height < 2:
        LOGGER.warning(f"      ⚠️ Grille trop petite ({width}x{height}). Vérifiez la bbox.")
        return None, None, None, (0, 0)

    grid_x = np.linspace(x_min, x_max, width)
    grid_y = np.linspace(y_max, y_min, height)  # y_max d'abord (nord en haut)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    # 4. Transformer les coordonnées du swath en UTM
    lon_flat = lon_2d.flatten()
    lat_flat = lat_2d.flatten()
    data_flat = data_2d.flatten()

    # Masquer les NaN et les valeurs aberrantes
    mask_valid = np.isfinite(data_flat) & np.isfinite(lon_flat) & np.isfinite(lat_flat)
    # Filtrer les pixels hors de la bbox (± marge)
    margin = 0.5  # degrés de marge
    mask_valid &= (lon_flat >= minlon - margin) & (lon_flat <= maxlon + margin)
    mask_valid &= (lat_flat >= minlat - margin) & (lat_flat <= maxlat + margin)

    if np.sum(mask_valid) < 10:
        LOGGER.warning(f"      ⚠️ Pas assez de pixels valides dans la bbox ({np.sum(mask_valid)}).")
        return None, None, None, (0, 0)

    x_swath, y_swath = transformer.transform(lon_flat[mask_valid], lat_flat[mask_valid])
    data_valid = data_flat[mask_valid]

    # 5. Interpolation du swath irrégulier sur la grille régulière
    points = np.column_stack([x_swath, y_swath])
    grid_data = griddata(points, data_valid, (grid_xx, grid_yy), method=method)

    # 6. Construire le transform rasterio
    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    return grid_data, transform, crs_str, (height, width)


def _save_as_tif(data_2d, transform, crs_str, output_path, nodata=np.nan):
    """Sauvegarde une matrice 2D en GeoTIFF."""
    height, width = data_2d.shape
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=height, width=width, count=1,
        dtype='float32', crs=crs_str,
        transform=transform, nodata=nodata
    ) as dst:
        dst.write(data_2d.astype(np.float32), 1)


# ============================================================
# TRAITEMENT DE LA LST SLSTR (THERMIQUE ~1 KM)
# ============================================================

def process_slstr_lst(item_dict, dossier_sortie, prefixe_nom, bbox_lonlat):
    """
    Télécharge, reprojette et exporte la LST SLSTR en TIF.

    Retourne le chemin du TIF généré, ou None en cas d'échec.
    """
    tif_path = os.path.join(dossier_sortie, f"{prefixe_nom}_LST_S3_1km.tif")
    if os.path.exists(tif_path):
        LOGGER.info(f"      ✅ LST S3 déjà traitée : {tif_path}")
        return tif_path

    assets = item_dict.get("assets", {})
    if "lst-in" not in assets or "slstr-geodetic-in" not in assets:
        LOGGER.warning("      ⚠️ Assets LST ou géodésiques manquants.")
        return None

    tmp_dir = os.path.join(dossier_sortie, "_tmp_nc")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # 1. Télécharger les NetCDF
        nc_lst = download_netcdf_asset(assets["lst-in"]["href"],
                                       os.path.join(tmp_dir, f"{prefixe_nom}_lst.nc"))
        nc_geo = download_netcdf_asset(assets["slstr-geodetic-in"]["href"],
                                       os.path.join(tmp_dir, f"{prefixe_nom}_geodetic.nc"))

        # 2. Lire les données
        ds_lst = xr.open_dataset(nc_lst)
        ds_geo = xr.open_dataset(nc_geo)

        lst_raw = ds_lst['LST'].values.squeeze()          # Kelvin
        lat_2d = ds_geo['latitude_in'].values.squeeze()
        lon_2d = ds_geo['longitude_in'].values.squeeze()

        ds_lst.close()
        ds_geo.close()

        # Convertir K → °C
        lst_celsius = np.where((lst_raw > 200) & (lst_raw < 350), lst_raw - 273.15, np.nan)

        LOGGER.info(f"      🌡️ LST brute S3 : {np.nanmin(lst_celsius):.1f}°C – {np.nanmax(lst_celsius):.1f}°C")

        # 3. Reprojection swath → grille régulière 1km
        grid_data, transform, crs_str, (h, w) = _reproject_swath_to_grid(
            lst_celsius, lon_2d, lat_2d, bbox_lonlat,
            target_res_m=1000, method='nearest'
        )

        if grid_data is None:
            return None

        nb_valid = np.count_nonzero(~np.isnan(grid_data))
        pct = (nb_valid / grid_data.size) * 100 if grid_data.size > 0 else 0
        LOGGER.info(f"      📊 Pixels valides après reprojection : {nb_valid}/{grid_data.size} ({pct:.0f}%)")

        if pct < 5:
            LOGGER.warning(f"      ❌ Trop peu de pixels valides, image ignorée.")
            return None

        # 4. Sauvegarder en TIF
        _save_as_tif(grid_data, transform, crs_str, tif_path)
        LOGGER.info(f"      ✅ LST S3 sauvegardée : {tif_path} ({h}x{w})")
        return tif_path

    except Exception as e:
        LOGGER.error(f"      ❌ Erreur traitement SLSTR LST : {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return None


# ============================================================
# TRAITEMENT DES BANDES OPTIQUES SYNERGY (~300 M)
# ============================================================

# Mapping des bandes Synergy → longueurs d'onde et usages
# Les bandes OLCI (OaXX) sont à 300m, les bandes SLSTR (SXX) aussi dans le produit Synergy
BANDES_SYNERGY = {
    "syn-oa04-reflectance": {"var": "SDR_Oa04", "nom": "Blue",  "lambda_nm": 490},  # Bleu
    "syn-oa06-reflectance": {"var": "SDR_Oa06", "nom": "Red",   "lambda_nm": 681},  # Rouge
    "syn-oa08-reflectance": {"var": "SDR_Oa08", "nom": "NIR",   "lambda_nm": 865},  # Proche IR
    "syn-oa17-reflectance": {"var": "SDR_Oa17", "nom": "NIR2",  "lambda_nm": 865},  # NIR2
    "syn-s5n-reflectance":  {"var": "SDR_S5n",  "nom": "SWIR1", "lambda_nm": 1613}, # SWIR (nadir)
}


def process_synergy_optique(item_dict, dossier_sortie, prefixe_nom, bbox_lonlat):
    """
    Télécharge les bandes optiques Synergy, les reprojette à 300m, calcule les
    indices spectraux (NDVI, NDWI, SAVI, EVI), et exporte chaque résultat en TIF.

    Retourne un dict {nom_indice: chemin_tif} ou None en cas d'échec.
    """
    assets = item_dict.get("assets", {})

    if "geolocation" not in assets:
        LOGGER.warning("      ⚠️ Asset de géolocalisation Synergy manquant.")
        return None

    tmp_dir = os.path.join(dossier_sortie, "_tmp_nc")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        # 1. Télécharger la géolocalisation
        nc_geo = download_netcdf_asset(
            assets["geolocation"]["href"],
            os.path.join(tmp_dir, f"{prefixe_nom}_geo_syn.nc")
        )
        ds_geo = xr.open_dataset(nc_geo)
        lat_syn = ds_geo['lat'].values.squeeze()
        lon_syn = ds_geo['lon'].values.squeeze()
        ds_geo.close()

        # 2. Télécharger et reprojeter chaque bande optique
        bandes_reprojetees = {}  # nom -> np.ndarray 2D
        ref_transform = None
        ref_crs = None
        ref_shape = None

        for asset_key, info in BANDES_SYNERGY.items():
            if asset_key not in assets:
                LOGGER.info(f"      ⏭️  Bande {info['nom']} ({asset_key}) non disponible, skip.")
                continue

            nc_path = download_netcdf_asset(
                assets[asset_key]["href"],
                os.path.join(tmp_dir, f"{prefixe_nom}_{info['nom']}.nc")
            )

            ds_band = xr.open_dataset(nc_path)
            var_name = info["var"]
            if var_name not in ds_band:
                # Essayer le nom sans n/o suffixe
                alt_names = [v for v in ds_band.data_vars if var_name.replace('n', '').replace('o', '') in v.replace('n', '').replace('o', '')]
                if alt_names:
                    var_name = alt_names[0]
                else:
                    LOGGER.warning(f"      ⚠️ Variable {info['var']} non trouvée dans {nc_path}")
                    ds_band.close()
                    continue

            band_data = ds_band[var_name].values.squeeze()
            ds_band.close()

            # Masquer les valeurs non physiques (réflectance doit être entre 0 et 1)
            band_data = np.where((band_data >= 0) & (band_data <= 1.0), band_data, np.nan)

            grid_data, transform, crs_str, shape = _reproject_swath_to_grid(
                band_data, lon_syn, lat_syn, bbox_lonlat,
                target_res_m=300, method='nearest'
            )

            if grid_data is not None:
                bandes_reprojetees[info["nom"]] = grid_data
                if ref_transform is None:
                    ref_transform = transform
                    ref_crs = crs_str
                    ref_shape = shape
                LOGGER.info(f"      ✅ Bande {info['nom']} ({info['lambda_nm']}nm) reprojetée : {shape[1]}x{shape[0]}")

        if not bandes_reprojetees:
            LOGGER.error("      ❌ Aucune bande optique reprojetée avec succès.")
            return None

        # 3. Sauvegarder les bandes brutes en TIF
        tif_paths = {}
        for nom_bande, data in bandes_reprojetees.items():
            tif_path = os.path.join(dossier_sortie, f"{prefixe_nom}_S3_{nom_bande}.tif")
            _save_as_tif(data, ref_transform, ref_crs, tif_path)
            tif_paths[nom_bande] = tif_path

        # 4. Calculer les indices spectraux
        if "Red" in bandes_reprojetees and "NIR" in bandes_reprojetees:
            red = bandes_reprojetees["Red"]
            nir = bandes_reprojetees["NIR"]

            # NDVI
            ndvi = np.divide(nir - red, nir + red,
                           out=np.full_like(red, np.nan), where=(nir + red) != 0)
            ndvi_path = os.path.join(dossier_sortie, f"{prefixe_nom}_S3_NDVI.tif")
            _save_as_tif(ndvi, ref_transform, ref_crs, ndvi_path)
            tif_paths["NDVI"] = ndvi_path
            LOGGER.info(f"      📊 NDVI S3 : [{np.nanmin(ndvi):.2f}, {np.nanmax(ndvi):.2f}]")

            # SAVI
            L = 0.5
            savi = np.divide((nir - red) * (1 + L), nir + red + L,
                           out=np.full_like(red, np.nan), where=(nir + red + L) != 0)
            savi_path = os.path.join(dossier_sortie, f"{prefixe_nom}_S3_SAVI.tif")
            _save_as_tif(savi, ref_transform, ref_crs, savi_path)
            tif_paths["SAVI"] = savi_path

            # EVI (si Blue dispo)
            if "Blue" in bandes_reprojetees:
                blue = bandes_reprojetees["Blue"]
                denom_evi = nir + 6.0 * red - 7.5 * blue + 1.0
                evi = np.divide(2.5 * (nir - red), denom_evi,
                              out=np.full_like(red, np.nan), where=denom_evi != 0)
                evi_path = os.path.join(dossier_sortie, f"{prefixe_nom}_S3_EVI.tif")
                _save_as_tif(evi, ref_transform, ref_crs, evi_path)
                tif_paths["EVI"] = evi_path

        # NDWI (Green/NIR) - On utilise OA06 comme proxy du Green (pas parfait mais acceptable)
        if "Blue" in bandes_reprojetees and "NIR" in bandes_reprojetees:
            green_proxy = bandes_reprojetees["Blue"]  # OA04 (490nm) est proche du vert
            nir = bandes_reprojetees["NIR"]
            ndwi = np.divide(green_proxy - nir, green_proxy + nir,
                           out=np.full_like(nir, np.nan), where=(green_proxy + nir) != 0)
            ndwi_path = os.path.join(dossier_sortie, f"{prefixe_nom}_S3_NDWI.tif")
            _save_as_tif(ndwi, ref_transform, ref_crs, ndwi_path)
            tif_paths["NDWI"] = ndwi_path

        LOGGER.info(f"      ✅ {len(tif_paths)} fichiers TIF générés pour Synergy Optique.")
        return tif_paths

    except Exception as e:
        LOGGER.error(f"      ❌ Erreur traitement Synergy Optique : {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return None
