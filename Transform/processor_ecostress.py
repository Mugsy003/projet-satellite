import re
import numpy as np
import odc.stac
from config import LOGGER

def rename_ecostress_assets(item):
    """
    Renomme dynamiquement les clés d'assets STAC NASA
    vers des noms de bandes simples : LST, cloud, water, etc.
    Ne conserve que les liens directs HTTPS (ignore les clés 's3_').
    
    Les clés STAC NASA ont le format:
      002/.../ECOv002_L2T_LSTE_..._LST.tif
    On extrait le suffixe depuis le href (nom du fichier .tif) pour identifier la bande.
    """
    new_assets = {}
    for key, asset in list(item.assets.items()):
        if key.startswith("s3_") or "metadata" in key:
            continue
            
        if key == "browse" or "thumbnail" in key:
            new_assets[key] = asset
            continue
        
        # Extraire le nom court depuis le href (plus fiable que la clé)
        # Ex: ...ECOv002_L2T_LSTE_39156_006_32UPB_20250602T121900_0713_01_LST.tif -> LST
        href = asset.href if hasattr(asset, 'href') else ""
        filename = href.split("/")[-1].replace(".tif", "")
        
        # Le suffixe est après le dernier "_01_" (fin de l'identifiant standard)
        parts = filename.split("_")
        # Les noms de fichiers finissent par: ..._0713_01_LST ou ..._0713_01_LST_err
        # On cherche le suffixe après "01"
        try:
            idx_01 = len(parts) - 1
            for j in range(len(parts) - 1, -1, -1):
                if parts[j] == "01":
                    idx_01 = j
                    break
            short_name = "_".join(parts[idx_01 + 1:])
        except:
            short_name = parts[-1]
        
        if short_name:
            new_assets[short_name] = asset
        
    item.assets = new_assets
    return item

def _extract_epsg_from_item(item):
    """
    Extrait le code EPSG depuis le code MGRS contenu dans l'ID de l'item.
    Ex: ECOv002_L2T_LSTE_36835_009_32UPB_20250104T011432_0713_01
        -> MGRS = 32UPB -> UTM zone 32N -> EPSG:32632
    """
    match = re.search(r'_(\d{2}[A-Z]{3})_', item.id)
    if match:
        mgrs = match.group(1)
        utm_zone = int(mgrs[:2])
        hemisphere = 'N' if mgrs[2] >= 'N' else 'S'
        epsg = 32600 + utm_zone if hemisphere == 'N' else 32700 + utm_zone
        return f"EPSG:{epsg}"
    return "EPSG:4326"  # Fallback en WGS84

def process_ecostress_timeseries(mes_items, bbox):
    """
    Traite la série temporelle ECOSTRESS.
    Charge uniquement la bande LST et retourne les données en °C.
    """
    mes_items = sorted(mes_items, key=lambda x: x.datetime)
    
    # On modifie les clés d'assets de chaque item pour odc-stac
    for item in mes_items:
        rename_ecostress_assets(item)
        
    images_finales = [] 
    LOGGER.info(f"   🚀 Lancement du traitement ECOSTRESS ({len(mes_items)} dates)...")
    
    for i, item in enumerate(mes_items):
        anchor_date = item.datetime
        date_str = anchor_date.strftime("%Y-%m-%d_%Hh%M")
        
        LOGGER.info(f"\n   📅 [{i+1}/{len(mes_items)}] Image ECOSTRESS du {date_str}")
        
        if "LST" not in item.assets:
            LOGGER.warning(f"      ⚠️ Pas d'asset LST trouvé, passage à l'image suivante.")
            continue
        
        try:
            # Extraire le CRS depuis le code MGRS dans l'ID
            crs = _extract_epsg_from_item(item)
            
            # Charger uniquement la bande LST - résolution ~70m
            local_cube = odc.stac.stac_load(
                [item], 
                bands=["LST"], 
                bbox=bbox, 
                chunks={}, 
                resolution=70,
                crs=crs
            )
            
            if "LST" not in local_cube:
                LOGGER.warning("      ⚠️ LST non trouvée dans le cube, passage à l'image suivante.")
                continue

            lst_raw = local_cube["LST"].isel(time=0).values.astype(float)
            
            # ECOSTRESS L2T LSTE v002 : LST stockée directement en Kelvin (float32)
            # Pas de scale factor - conversion directe K -> °C
            lst_celsius = np.where(lst_raw > 200, lst_raw - 273.15, np.nan)
            
            # Vérification de la validité des données
            nb_valid = np.count_nonzero(~np.isnan(lst_celsius))
            total = lst_celsius.size
            pct_valid = (nb_valid / total) * 100 if total > 0 else 0
            LOGGER.info(f"      📊 Pixels valides : {nb_valid}/{total} ({pct_valid:.1f}%)")
            
            if nb_valid > 0:
                LOGGER.info(f"      🌡️ LST : {np.nanmin(lst_celsius):.1f}°C - {np.nanmax(lst_celsius):.1f}°C")
            
            if pct_valid < 5:
                LOGGER.warning(f"      ❌ Trop peu de pixels valides, image ignorée.")
                continue
            
            empreinte_transform = local_cube.odc.geobox.transform
            empreinte_crs = local_cube.odc.geobox.crs.to_wkt()
            
            images_finales.append({
                "date": date_str,
                "lst_celsius": lst_celsius,
                "transform": empreinte_transform,
                "crs": empreinte_crs
            })
            LOGGER.info(f"      ✅ Image traitée avec succès.")
            
        except Exception as e:
            LOGGER.error(f"      ⚠️ Échec critique ECOSTRESS sur la date {date_str}. Erreur: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            continue
            
    if not images_finales:
        LOGGER.warning("   ⚠️ Aucune image ECOSTRESS traitée avec succès.")
        return None

    return images_finales
