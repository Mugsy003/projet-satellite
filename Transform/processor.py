import numpy as np
import odc.stac
import planetary_computer as pc
import Transform.indices as indices
from Utils import get_landsat_mask, landsat_dn_to_reflectance, filtre_median_inteligent, calcul_couverture
from config import LOGGER

# =====================================================================
# SOUS-FONCTIONS 
# =====================================================================

def _filtrer_doublons_journaliers_metadata(mes_items):
    """Ne conserve que la meilleure image par jour (basé uniquement sur les métadonnées)."""
    LOGGER.info("   -> Filtrage des doublons (Conservation de la meilleure image via métadonnées)...")
    meilleurs_indices = {}
        
    for i, item in enumerate(mes_items):
        date_str = item.datetime.strftime("%Y-%m-%d")
        clouds = item.properties.get('eo:cloud_cover', 100)
        score = -clouds # Moins il y a de nuages, meilleur est le score
            
        if date_str not in meilleurs_indices or score > meilleurs_indices[date_str]["score"]:
            meilleurs_indices[date_str] = {"index": i, "score": score}

    indices_uniques = sorted([val["index"] for val in meilleurs_indices.values()])
    mes_items_filtres = [mes_items[i] for i in indices_uniques]
    
    return mes_items_filtres


def _chercher_voisins_minimal(i, anchor_date, mes_items, max_jours, max_nuages):
    """Cherche des images proches dans le temps via les métadonnées (sans télécharger)."""
    indices = [i] # L'ancre est toujours en première position (index 0 de la future liste)
    for j, item in enumerate(mes_items):
        if i == j: continue
        diff = abs((item.datetime - anchor_date).days)
        clouds = item.properties.get('eo:cloud_cover', 100)
        if diff <= max_jours and clouds <= max_nuages:
            indices.append(j)
    return indices


# =====================================================================
# FONCTION PRINCIPALE 
# =====================================================================

def process_satellite_timeseries(mes_items, bbox, bands_of_interest, max_jours_fusion=3, max_nuages_rejet=60, min_couv_rejet=40, couverture_parfaite=95):
    
    mes_items = sorted(mes_items, key=lambda x: x.datetime)
    
    # 1. Nettoyage initial ultra-rapide (sans téléchargement)
    mes_items = _filtrer_doublons_journaliers_metadata(mes_items)
    
    images_finales = [] 
    LOGGER.info(f"   🚀 Lancement du traitement SÉQUENTIEL ({len(mes_items)} dates à analyser)...")
    
    # 2. Évaluation de chaque date UNE PAR UNE
    for i, anchor_item in enumerate(mes_items):
        anchor_date = anchor_item.datetime
        date_str = anchor_date.strftime("%Y-%m-%d_%Hh%M")
        clouds = anchor_item.properties.get('eo:cloud_cover', 100)
        
        # --- PRÉ-FILTRAGE METADONNÉES ---
        # Si c'est trop nuageux, on ne télécharge même pas !
        if clouds > max_nuages_rejet:
            LOGGER.info(f"\n   📅 {date_str} - ❌ REJETÉE (Métadonnées) : Nuages ({clouds:.1f}%) > {max_nuages_rejet}%")
            continue

        LOGGER.info(f"\n   📅 Évaluation de l'image du {date_str} (Nuages métadonnées: {clouds:.1f}%)")
        
        try:
            # --- CHARGEMENT INDIVIDUEL (ROBUSTE) ---
            # On identifie les voisins et on ne signe/télécharge QUE ce petit groupe
            indices_voisins = _chercher_voisins_minimal(i, anchor_date, mes_items, max_jours_fusion, max_nuages_rejet)
            items_a_charger = [pc.sign(mes_items[idx]) for idx in indices_voisins]
            
            LOGGER.info(f"      📥 Chargement de {len(items_a_charger)} image(s) pour cette date...")
            
            # chunks={} force le téléchargement direct en mémoire sans Dask, évitant les timeouts
            local_cube = odc.stac.stac_load(items_a_charger, bands=bands_of_interest, bbox=bbox, chunks={})
            
            # L'image "ancre" (celle du jour évalué) est TOUJOURS à l'index temporel 0 dans ce petit cube
            couverture = calcul_couverture(local_cube.isel(time=0))
            LOGGER.info(f"      📊 Couverture réelle calculée : {couverture:.1f}%")
            
            # --- DÉCISION FINALE ---
            if couverture < min_couv_rejet:
                LOGGER.info(f"      ❌ REJETÉE : Qualité spatiale insuffisante.")
                continue
                
            elif couverture >= couverture_parfaite and clouds < 10: 
                LOGGER.info(f"      ✅ PARFAITE : Conservée sans modification.")
                indices_a_fusionner_local = [0]
                etat = "Parfaite" 
                
            else:
                LOGGER.info(f"      🩹 Amélioration nécessaire. Utilisation de {len(indices_voisins)} image(s).")
                # On utilise toutes les images téléchargées dans le local_cube
                indices_a_fusionner_local = list(range(len(indices_voisins)))
                etat = "Reparee" if len(indices_voisins) > 1 else "Non_Reparee"

            # --- TRAITEMENT SPATIAL ---
            cube_a_traiter = local_cube.isel(time=indices_a_fusionner_local)
            qa_mask = get_landsat_mask(cube_a_traiter["qa_pixel"])
            masked_cube = cube_a_traiter.where(qa_mask > 0, np.nan)
            masked_cube = masked_cube.where(masked_cube != 0, np.nan)
            
            mosaic_data = masked_cube.squeeze("time") if len(indices_a_fusionner_local) == 1 else filtre_median_inteligent(masked_cube)
            
            # --- EXTRACTION RGB ET INDICES ---
            # L'image brute correspond toujours au temps 0 du cube local
            img_brute = local_cube[["red", "green", "blue"]].isel(time=0).to_array().values.transpose(1, 2, 0)
            img_reflectance_rgb = landsat_dn_to_reflectance(mosaic_data[["red", "green", "blue"]].to_array().values)
            
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
                "NDVI": ndvi_array,
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
                "nb_images_fusionnees": len(indices_a_fusionner_local),
                "etat": etat,
                "transform": empreinte_transform,
                "crs": empreinte_crs
            })
            
        except Exception as e:
            LOGGER.error(f"      ⚠️ Échec critique sur la date {date_str}, passage à la suivante. Erreur: {e}")
            continue
            
    if not images_finales:
        LOGGER.warning("   ⚠️ Aucune image n'a survécu au filtrage pour cette période.")
        return None

    return images_finales