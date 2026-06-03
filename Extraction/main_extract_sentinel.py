"""
Extraction/main_extract_sentinel.py

Extraction Sentinel-2 PILOTÉE PAR LE MANIFESTE LANDSAT.
Pour chaque image Landsat du manifeste, on cherche s'il existe une image
Sentinel-2 quasi-simultanée (± TIME_MARGIN_MINUTES). Cela garantit que
chaque image S2 extraite aura un partenaire Landsat thermique pour la fusion.
"""
import os
import json
import time
from datetime import timedelta

import pystac_client
import planetary_computer
from pystac_client.exceptions import APIError

from config import (LOGGER, SITES_PILOTES, OUTPUT_DIR, PREVIEWS_DIR,
                    TIME_MARGIN_MINUTES, ltd, radius_km)
from Utils import get_bbox_from_point


def load_landsat_manifest(output_dir):
    """Charge le manifeste Landsat (manifest_extraction.json)."""
    path = os.path.join(output_dir, "manifest_extraction.json")
    if not os.path.exists(path):
        LOGGER.error(f"Manifeste Landsat introuvable : {path}")
        LOGGER.error("Lancez d'abord  python -m Extraction.main_extract")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_landsat_datetimes(catalog, landsat_ids):
    """
    Interroge le catalogue STAC pour récupérer le datetime exact
    de chaque image Landsat à partir de ses IDs.
    Retourne une liste de tuples (landsat_id, datetime).
    """
    max_retries = 5
    for attempt in range(max_retries):
        try:
            search = catalog.search(collections=["landsat-c2-l2"], ids=landsat_ids)
            items = list(search.items())
            result = [(item.id, item.datetime) for item in items if item.datetime]
            return result
        except APIError as e:
            LOGGER.warning(f"   Erreur API STAC Landsat: {e}. Tentative {attempt+1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
        except Exception as e:
            LOGGER.error(f"   Erreur inattendue Landsat: {e}")
            return []
    return []


def search_s2_for_landsat_date(catalog, bbox, landsat_dt, margin_minutes, max_cloud):
    """
    Cherche une image Sentinel-2 dans une fenêtre temporelle autour de
    l'heure d'acquisition Landsat.
    Retourne la meilleure image S2 (plus faible couverture nuageuse) ou None.
    """
    dt_start = landsat_dt - timedelta(minutes=margin_minutes)
    dt_end = landsat_dt + timedelta(minutes=margin_minutes)
    time_range = f"{dt_start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{dt_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=time_range
            )
            items = list(search.items())
            break
        except APIError as e:
            LOGGER.warning(f"      Erreur API S2: {e}. Tentative {attempt+1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                return None
        except Exception as e:
            LOGGER.error(f"      Erreur inattendue S2: {e}")
            return None

    if not items:
        return None

    # Filtrer par couverture nuageuse
    items_propres = [it for it in items if it.properties.get("eo:cloud_cover", 100) <= max_cloud]

    if not items_propres:
        return None

    # Trier par proximité temporelle, puis par couverture nuageuse
    items_propres.sort(key=lambda it: (
        abs((it.datetime - landsat_dt).total_seconds()),
        it.properties.get("eo:cloud_cover", 100)
    ))

    return items_propres[0]


def main():
    LOGGER.info("DEBUT DE LA PHASE 1 SENTINEL-2 : EXTRACTION (pilotee par Landsat)")

    # 1. Charger le manifeste Landsat
    manifest_landsat = load_landsat_manifest(OUTPUT_DIR)
    if manifest_landsat is None:
        return

    # 2. Connexion au catalogue STAC
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

    manifest_extraction_s2 = {}
    stats_global = {"total_landsat": 0, "paires_trouvees": 0, "paires_rejetees_nuages": 0}

    # 3. Pour chaque site, récupérer les datetimes Landsat et chercher les S2 correspondants
    for pays, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"Site : {pays}")

        landsat_ids = manifest_landsat.get(pays, [])
        if not landsat_ids:
            LOGGER.warning(f"   Aucune image Landsat dans le manifeste pour {pays}.")
            continue

        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km)

        # Récupérer les datetimes exacts depuis le catalogue STAC
        LOGGER.info(f"   Recuperation des datetimes de {len(landsat_ids)} images Landsat...")
        landsat_dates = get_landsat_datetimes(catalog, landsat_ids)
        LOGGER.info(f"   {len(landsat_dates)} datetimes Landsat resolus.")

        s2_ids_for_site = []
        stats_global["total_landsat"] += len(landsat_dates)

        for landsat_id, landsat_dt in landsat_dates:
            LOGGER.info(f"   Landsat {landsat_id} ({landsat_dt.strftime('%Y-%m-%d %Hh%M')}) -> recherche S2 +/-{TIME_MARGIN_MINUTES} min...")

            best_s2 = search_s2_for_landsat_date(
                catalog, bbox, landsat_dt, TIME_MARGIN_MINUTES, ltd
            )

            if best_s2:
                s2_dt = best_s2.datetime
                delta = abs((s2_dt - landsat_dt).total_seconds()) / 60
                clouds = best_s2.properties.get("eo:cloud_cover", -1)
                LOGGER.info(f"      PAIRE TROUVEE : {best_s2.id} (delta={delta:.0f} min, nuages={clouds:.1f}%)")
                s2_ids_for_site.append(best_s2.id)
                stats_global["paires_trouvees"] += 1
            else:
                LOGGER.info(f"      Pas de S2 dans la fenetre.")

        # Dédupliquer (une même S2 peut matcher 2 Landsat si les orbites se chevauchent)
        s2_ids_for_site = list(dict.fromkeys(s2_ids_for_site))
        manifest_extraction_s2[pays] = s2_ids_for_site

        LOGGER.info(f"   Resultat {pays} : {len(s2_ids_for_site)} images S2 selectionnees pour {len(landsat_dates)} Landsat.")

    # 4. Sauvegarde du manifeste S2
    path_extract = os.path.join(OUTPUT_DIR, "manifest_extraction_s2.json")
    with open(path_extract, "w", encoding="utf-8") as f:
        json.dump(manifest_extraction_s2, f, indent=4)

    LOGGER.info(f"\nPHASE 1 SENTINEL-2 TERMINEE.")
    LOGGER.info(f"   Manifeste S2 sauvegarde : {path_extract}")
    LOGGER.info(f"   Stats : {stats_global['total_landsat']} Landsat analysees | {stats_global['paires_trouvees']} paires S2 trouvees")


if __name__ == "__main__":
    main()
