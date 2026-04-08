import numpy as np
import odc.stac
import Transform.indices as indices
from Utils import get_landsat_mask, landsat_dn_to_reflectance, filtre_median_inteligent, calcul_couverture
from config import LOGGER

# =====================================================================
# SOUS-FONCTIONS 
# =====================================================================

def _filtrer_doublons_journaliers(mes_items, data_cube):
    """Ne conserve que la meilleure image s'il y a plusieurs passages le même jour."""
    LOGGER.info("   -> Filtrage des doublons (Conservation de la meilleure image par jour)...")
    meilleurs_indices = {}
        
    for i, item in enumerate(mes_items):
        date_str = item.datetime.strftime("%Y-%m-%d")
        clouds = item.properties.get('eo:cloud_cover', 100)
        couverture = calcul_couverture(data_cube.isel(time=i))
        score = couverture - clouds 
            
        if date_str not in meilleurs_indices or score > meilleurs_indices[date_str]["score"]:
            if date_str in meilleurs_indices:
                LOGGER.info(f"      🔄 Doublon le {date_str} : Remplacement par une meilleure (Score: {score:.1f}).")
            meilleurs_indices[date_str] = {"index": i, "score": score}

    indices_uniques = sorted([val["index"] for val in meilleurs_indices.values()])
    mes_items_filtres = [mes_items[i] for i in indices_uniques]
    data_cube_filtre = data_cube.isel(time=indices_uniques)
    
    return mes_items_filtres, data_cube_filtre


def _chercher_voisins(i, anchor_date, mes_items, data_cube, max_jours, max_nuages, min_couv):
    """Cherche des images proches dans le temps pour réparer les trous."""
    indices_a_fusionner = [i]
    for j, voisin_item in enumerate(mes_items):
        if i == j: continue 
        
        diff_jours = abs((voisin_item.datetime - anchor_date).days)
        v_clouds = voisin_item.properties.get('eo:cloud_cover', 100)
        v_couv = calcul_couverture(data_cube.isel(time=j))
        
        if diff_jours <= max_jours and v_clouds <= max_nuages and v_couv >= min_couv:
            indices_a_fusionner.append(j)
    return indices_a_fusionner


# =====================================================================
# FONCTION PRINCIPALE 
# =====================================================================

def process_satellite_timeseries(mes_items, bbox, bands_of_interest, max_jours_fusion=3, max_nuages_rejet=60, min_couv_rejet=40, couverture_parfaite=95):
    
    mes_items = sorted(mes_items, key=lambda x: x.datetime)
    LOGGER.info("   -> Chargement du cube de données en mémoire...")
    data_cube = odc.stac.stac_load(mes_items, bands=bands_of_interest, bbox=bbox)
    
    # 1. Nettoyage initial
    mes_items, data_cube = _filtrer_doublons_journaliers(mes_items, data_cube)
    
    images_finales = [] 
    
    # 2. Évaluation de chaque date
    for i, anchor_item in enumerate(mes_items):
        anchor_date = anchor_item.datetime
        date_str = anchor_date.strftime("%Y-%m-%d")
        clouds = anchor_item.properties.get('eo:cloud_cover', 100)
        couverture = calcul_couverture(data_cube.isel(time=i))
        
        LOGGER.info(f"\n   📅 Évaluation de l'image du {date_str} (Nuages: {clouds:.1f}%, Couv: {couverture:.1f}%)")
        
        # --- DÉCISION ---
        if clouds > max_nuages_rejet or couverture < min_couv_rejet:
            LOGGER.info(f"      ❌ REJETÉE : Qualité insuffisante.")
            continue
            
        elif couverture >= couverture_parfaite and clouds < 10: 
            LOGGER.info(f"      ✅ PARFAITE : Conservée sans modification.")
            indices_a_fusionner = [i]
            etat = "Parfaite" 
            
        else:
            LOGGER.info(f"      🩹  Améliorée : Recherche de voisins à ±{max_jours_fusion} jours pour réparer...")
            indices_a_fusionner = _chercher_voisins(i, anchor_date, mes_items, data_cube, max_jours_fusion, max_nuages_rejet, min_couv_rejet)
            
            if len(indices_a_fusionner) == 1:
                LOGGER.warning(f"      ⚠️ ÉCHEC : Aucun voisin trouvé. L'image restera incomplète.")
                etat = "Non_Reparee"
            else:
                LOGGER.info(f"      🧩 Fusion de {len(indices_a_fusionner)} images au total pour réparer le {date_str}.")
                etat = "Reparee"

        # --- TRAITEMENT SPATIAL ---
        local_cube = data_cube.isel(time=indices_a_fusionner)
        qa_mask = get_landsat_mask(local_cube["qa_pixel"])
        masked_cube = local_cube.where(qa_mask > 0, np.nan)
        masked_cube = masked_cube.where(masked_cube != 0, np.nan)
        
        mosaic_data = masked_cube.squeeze("time") if len(indices_a_fusionner) == 1 else filtre_median_inteligent(masked_cube)
        
        # --- EXTRACTION RGB ---
        img_brute = data_cube[["red", "green", "blue"]].isel(time=i).to_array().values.transpose(1, 2, 0)
        img_reflectance_rgb = landsat_dn_to_reflectance(mosaic_data[["red", "green", "blue"]].to_array().values)
        
        # CALCUL DES INDICES DE VÉGÉTATION ET URBAINS
        red_ref = landsat_dn_to_reflectance(mosaic_data["red"].values)
        green_ref = landsat_dn_to_reflectance(mosaic_data["green"].values)
        blue_ref = landsat_dn_to_reflectance(mosaic_data["blue"].values)
        nir_ref = landsat_dn_to_reflectance(mosaic_data["nir08"].values)
        swir_ref = landsat_dn_to_reflectance(mosaic_data["swir16"].values)

        thermal_dn = mosaic_data["lwir11"].values
        bt_kelvin = (thermal_dn * 0.00341802) + 149.0
        img_thermique_celsius = bt_kelvin - 273.15
        ndvi_array = indices.calculate_ndvi(red_ref, nir_ref)
        lst_array = indices.calculate_lst_step_by_step(bt_kelvin, ndvi_array)
        empreinte_transform = mosaic_data.odc.geobox.transform
        empreinte_crs = mosaic_data.odc.geobox.crs.to_wkt()

        indices_dict = {
            "NDVI": indices.calculate_ndvi(red_ref, nir_ref),
            "NDWI": indices.calculate_ndwi(green_ref, nir_ref),
            "NDBI": indices.calculate_ndbi(swir_ref, nir_ref),
            "EVI": indices.calculate_evi(red_ref, nir_ref, blue_ref),
            "SAVI": indices.calculate_savi(red_ref, nir_ref),
            "LST": lst_array,
            "Thermique_B10": img_thermique_celsius  
        }

        images_finales.append({
            "date": date_str,
            "brute": img_brute,
            "reflectance": img_reflectance_rgb,
            "indices": indices_dict,  
            "nb_images_fusionnees": len(indices_a_fusionner),
            "etat": etat,
            "transform": empreinte_transform,
            "crs": empreinte_crs
        })
        
    if not images_finales:
        LOGGER.warning("   ⚠️ Aucune image n'a survécu au filtrage pour cette période.")
        return None

    return images_finales