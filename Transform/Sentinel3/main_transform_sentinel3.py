"""
Transform/Sentinel3/main_transform_sentinel3.py
================================================
Orchestrateur de la transformation Sentinel-3.
Lit les manifestes JSON générés par l'extraction, puis pour chaque paire :
  1. Télécharge et reprojette la LST SLSTR (~1 km) en TIF
  2. Télécharge et reprojette les bandes optiques Synergy (~300 m) en TIF
  3. Calcule les indices spectraux (NDVI, SAVI, EVI, NDWI) à 300 m
"""
import os
import json
from config import LOGGER, SITES_PILOTES, OUTPUT_DIR, radius_km_s3, ltd
from Utils.geo import get_bbox_from_point
import Transform.Sentinel3.processor_sentinel3 as processor_s3


def main():
    LOGGER.info("========================================")
    LOGGER.info("🔥 DÉMARRAGE TRANSFORMATION SENTINEL-3")
    LOGGER.info("========================================")

    nb_images_total = 0
    nb_images_ok = 0

    for nom_site, coords in SITES_PILOTES.items():
        chemin_manifeste = os.path.join(OUTPUT_DIR, f"manifest_extraction_S3_{nom_site}.json")
        if not os.path.exists(chemin_manifeste):
            continue

        LOGGER.info(f"\n🌍 === Traitement du site : {nom_site} ===")
        
        # La bbox en lon/lat pour la reprojection
        bbox_lonlat = get_bbox_from_point(coords["lon"], coords["lat"], radius_km_s3)

        # Dossier de sortie pour ce site
        dossier_sortie = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{nom_site}_S3", "TIF_Data")
        os.makedirs(dossier_sortie, exist_ok=True)

        with open(chemin_manifeste, "r") as f:
            paires = json.load(f)

        LOGGER.info(f"   📋 {len(paires)} paires S3 (SLSTR + Synergy) à traiter.")

        # Liste des dates retenues (faible couverture) pour ce site
        dates_retenues = []

        for i, paire in enumerate(paires):
            date_str = paire["date"]
            item_t = paire["thermique"]
            item_o = paire["optique"]
            ecart = paire.get("ecart_minutes", "?")

            prefixe = f"{date_str}_{nom_site}"
            nb_images_total += 1

            LOGGER.info(f"\n   📅 [{i+1}/{len(paires)}] Paire du {date_str} (écart SLSTR/Synergy : {ecart if isinstance(ecart, str) else f'{ecart:.0f}'} min)")

            # Requête STAC pour avoir des tokens frais et propriétés à jour
            import pystac_client
            import planetary_computer
            catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
            
            id_t = item_t if isinstance(item_t, str) else item_t.get("id")
            id_o = item_o if isinstance(item_o, str) else item_o.get("id")
            
            search_t = catalog.search(collections=["sentinel-3-slstr-lst-l2-netcdf"], ids=[id_t])
            search_o = catalog.search(collections=["sentinel-3-synergy-syn-l2-netcdf"], ids=[id_o])
            
            items_t_fresh = list(search_t.items())
            items_o_fresh = list(search_o.items())
            
            if not items_t_fresh or not items_o_fresh:
                LOGGER.warning(f"      ⚠️ Impossible d'actualiser les items S3 pour {date_str}.")
                continue

            # Récupérer la couverture nuageuse depuis les propriétés fraîchement récupérées
            props_t = items_t_fresh[0].properties
            props_o = items_o_fresh[0].properties
            cc_t = props_t.get("eo:cloud_cover")
            cc_o = props_o.get("eo:cloud_cover")

            # Si l'une des deux images dépasse le seuil `ltd`, supprimer fichiers existants et skip
            seuil = ltd
            too_cloudy = False
            if cc_t is not None and cc_t > seuil:
                too_cloudy = True
            if cc_o is not None and cc_o > seuil:
                too_cloudy = True

            # Dossier contenant les TIF pour ce site
            # (déjà créé plus haut)

            if too_cloudy:
                LOGGER.info(f"      ☁️ Paire {date_str} rejetée (cloud SLSTR={cc_t}, Synergy={cc_o}) > {seuil}% — suppression fichiers existants.")
                # Supprimer fichiers avec ce préfixe dans le dossier de sortie
                try:
                    for fname in os.listdir(dossier_sortie):
                        if fname.startswith(prefixe):
                            path_rm = os.path.join(dossier_sortie, fname)
                            try:
                                if os.path.isdir(path_rm):
                                    import shutil
                                    shutil.rmtree(path_rm)
                                else:
                                    os.remove(path_rm)
                                LOGGER.info(f"         🗑️ Supprimé : {path_rm}")
                            except Exception as e:
                                LOGGER.warning(f"         ⚠️ Impossible de supprimer {path_rm}: {e}")
                except FileNotFoundError:
                    pass
                continue

            # Paire retenue
            dates_retenues.append(date_str)

            # Signer et traiter
            item_t_frais = planetary_computer.sign(items_t_fresh[0].to_dict())
            item_o_frais = planetary_computer.sign(items_o_fresh[0].to_dict())

            # 1. Traitement de la LST SLSTR (1 km)
            tif_lst = processor_s3.process_slstr_lst(item_t_frais, dossier_sortie, prefixe, bbox_lonlat)

            # 2. Traitement des bandes optiques Synergy (300 m) + indices
            tif_optique = processor_s3.process_synergy_optique(item_o_frais, dossier_sortie, prefixe, bbox_lonlat)

            if tif_lst and tif_optique:
                nb_images_ok += 1
                LOGGER.info(f"   ✅ Paire {date_str} traitée avec succès.")
            else:
                LOGGER.warning(f"   ⚠️ Paire {date_str} partiellement traitée (LST: {'OK' if tif_lst else 'FAIL'}, Optique: {'OK' if tif_optique else 'FAIL'}).")

        # Sauvegarder JSON des dates retenues pour ce site
        chemin_dates_retenues = os.path.join(OUTPUT_DIR, f"manifest_extraction_S3_{nom_site}_lowcloud_dates.json")
        try:
            with open(chemin_dates_retenues, 'w') as f_dates:
                json.dump(dates_retenues, f_dates, indent=2)
            LOGGER.info(f"   💾 Dates retenues (faible couverture) sauvegardées : {chemin_dates_retenues}")
        except Exception as e:
            LOGGER.warning(f"   ⚠️ Impossible de sauvegarder les dates retenues: {e}")

    LOGGER.info(f"\n{'='*50}")
    LOGGER.info(f"✅ TRANSFORMATION SENTINEL-3 TERMINÉE.")
    LOGGER.info(f"   {nb_images_ok}/{nb_images_total} paires traitées avec succès.")
    LOGGER.info(f"{'='*50}")


if __name__ == "__main__":
    main()
