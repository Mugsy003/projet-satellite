import numpy as np
import odc.stac
import planetary_computer as pc
import Transform.indices as indices
from Utils import filtre_median_inteligent
from config import LOGGER

def get_s2_mask(scl_array):
    """
    Crée un masque valide à partir de la couche SCL de Sentinel-2.
    Valeurs valides (souvent 4: Végétation, 5: Sol nu, 6: Eau, 7: Non classifié, 11: Neige)
    Nuages (8, 9, 10), Ombre (3), Défectueux (0, 1, 2)
    """
    valid_classes = [4, 5, 6, 7, 11]
    return np.isin(scl_array, valid_classes)

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
    indices = [i] # L'ancre est toujours en première position
    for j, item in enumerate(mes_items):
        if i == j: continue
        diff = abs((item.datetime - anchor_date).days)
        clouds = item.properties.get('eo:cloud_cover', 100)
        if diff <= max_jours and clouds <= max_nuages:
            indices.append(j)
    return indices

def process_s2_timeseries(mes_items, bbox, bands_of_interest, max_jours_fusion=3, max_nuages_rejet=60, min_couv_rejet=40, couverture_parfaite=95):
    
    mes_items = sorted(mes_items, key=lambda x: x.datetime)
    mes_items = _filtrer_doublons_journaliers_metadata(mes_items)
    
    images_finales = [] 
    LOGGER.info(f"   🚀 Lancement du traitement SENTINEL-2 ({len(mes_items)} dates)...")
    
    for i, anchor_item in enumerate(mes_items):
        anchor_date = anchor_item.datetime
        date_str = anchor_date.strftime("%Y-%m-%d_%Hh%M")
        clouds = anchor_item.properties.get('eo:cloud_cover', 100)
        
        if clouds > max_nuages_rejet:
            LOGGER.info(f"\n   📅 {date_str} - ❌ REJETÉE : Nuages ({clouds:.1f}%) > {max_nuages_rejet}%")
            continue

        LOGGER.info(f"\n   📅 Évaluation de l'image S2 du {date_str} (Nuages: {clouds:.1f}%)")
        
        try:
            indices_voisins = _chercher_voisins_minimal(i, anchor_date, mes_items, max_jours_fusion, max_nuages_rejet)
            items_a_charger = [pc.sign(mes_items[idx]) for idx in indices_voisins]
            
            # Résolution de 10m forcée pour Sentinel-2
            local_cube = odc.stac.stac_load(items_a_charger, bands=bands_of_interest, bbox=bbox, chunks={}, resolution=10)
            
            anchor_cube = local_cube.isel(time=0)
            anchor_mask = get_s2_mask(anchor_cube["SCL"].values)
            
            # La couverture réelle est le % de pixels qui sont à la fois dans l'image (>0) ET clairs (anchor_mask)
            total_pixels = anchor_cube["B04"].size
            pixels_valides = np.count_nonzero((anchor_cube["B04"].values > 0) & anchor_mask)
            couverture = (pixels_valides / total_pixels) * 100
            
            LOGGER.info(f"      📊 Couverture claire et réelle calculée localement : {couverture:.1f}%")
            
            if couverture < min_couv_rejet:
                LOGGER.info(f"      ❌ REJETÉE : Qualité spatiale insuffisante.")
                continue
                
            elif couverture >= couverture_parfaite: 
                LOGGER.info(f"      ✅ PARFAITE : Conservée sans modification.")
                indices_a_fusionner_local = [0]
                etat = "Parfaite" 
                
            else:
                LOGGER.info(f"      🩹 Amélioration nécessaire. Utilisation de {len(indices_voisins)} image(s).")
                indices_a_fusionner_local = list(range(len(indices_voisins)))
                etat = "Reparee" if len(indices_voisins) > 1 else "Non_Reparee"

            cube_a_traiter_brut = local_cube.isel(time=indices_a_fusionner_local)
            
            # Masque SCL pour tout le cube local
            qa_mask = get_s2_mask(cube_a_traiter_brut["SCL"].values) 
            
            # Application du filtre intelligent si réparée
            if etat == "Reparee":
                masked_cube = cube_a_traiter_brut.where(qa_mask)
                cube_a_traiter_final = filtre_median_inteligent(masked_cube)
                qa_mask_final = ~np.isnan(cube_a_traiter_final["B04"].values)
            else:
                cube_a_traiter_final = cube_a_traiter_brut.isel(time=0)
                qa_mask_final = qa_mask if qa_mask.ndim == 2 else qa_mask[0]

            # Extraction des données raw (sans mask) pour l'image brute, toujours basée sur l'ancre (time=0)
            anchor_cube = cube_a_traiter_brut.isel(time=0)
            raw_red = anchor_cube["B04"].values / 10000.0
            raw_green = anchor_cube["B03"].values / 10000.0
            raw_blue = anchor_cube["B02"].values / 10000.0

            # Remplace les 0.0 (nodata S2) par des NaN pour ne pas fausser l'étirement des couleurs
            raw_red = np.where(raw_red == 0.0, np.nan, raw_red)
            raw_green = np.where(raw_green == 0.0, np.nan, raw_green)
            raw_blue = np.where(raw_blue == 0.0, np.nan, raw_blue)

            # Extraction des données traitées
            red = np.where(qa_mask_final, cube_a_traiter_final["B04"].values / 10000.0, np.nan)
            green = np.where(qa_mask_final, cube_a_traiter_final["B03"].values / 10000.0, np.nan)
            blue = np.where(qa_mask_final, cube_a_traiter_final["B02"].values / 10000.0, np.nan)
            nir = np.where(qa_mask_final, cube_a_traiter_final["B08"].values / 10000.0, np.nan)
            swir = np.where(qa_mask_final, cube_a_traiter_final["B11"].values / 10000.0, np.nan)

            # Image RGB pour la visualisation
            img_rgb_hwc = np.stack([raw_red, raw_green, raw_blue], axis=-1) # (H, W, 3) pour l'image brute
            img_rgb_chw = np.stack([red, green, blue], axis=0)  # (3, H, W) pour l'image traitée

            # Calcul des indices
            ndvi_array = indices.calculate_ndvi(red, nir)
            
            empreinte_transform = cube_a_traiter_final.odc.geobox.transform
            empreinte_crs = cube_a_traiter_final.odc.geobox.crs.to_wkt()

            indices_dict = {
                "NDVI": ndvi_array,
                "NDWI": indices.calculate_ndwi(green, nir),
                "NDBI": indices.calculate_ndbi(swir, nir),
                "EVI": indices.calculate_evi(red, nir, blue),
                "SAVI": indices.calculate_savi(red, nir)
            }

            images_finales.append({
                "date": date_str,
                "brute": img_rgb_hwc,
                "reflectance": img_rgb_chw,
                "indices": indices_dict,  
                "nb_images_fusionnees": len(indices_a_fusionner_local),
                "etat": etat,
                "transform": empreinte_transform,
                "crs": empreinte_crs
            })
            
        except Exception as e:
            LOGGER.error(f"      ⚠️ Échec critique S2 sur la date {date_str}. Erreur: {e}")
            continue
            
    if not images_finales:
        LOGGER.warning("   ⚠️ Aucune image S2 n'a survécu au filtrage.")
        return None

    return images_finales
