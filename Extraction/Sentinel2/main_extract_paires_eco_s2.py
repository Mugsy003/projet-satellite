"""
Extraction/main_extract_paires_eco_s2.py

Extraction intelligente des paires Sentinel-2 et ECOSTRESS.
Stratégie :
1. Recherche des images Sentinel-2 sur la période définie (filtrées par nuages).
2. Pour chaque image S2, recherche d'une image ECOSTRESS dans un intervalle de temps (± TIME_MARGIN_MINUTES).
3. Si la paire existe, on conserve les deux IDs.
4. Génération des manifestes respectifs pour alimenter les scripts de transformation.
"""
import os
import json
from datetime import timedelta

import Extraction.Sentinel2.stac_client_sentinel as stac_s2
import Extraction.ECOSTRESS.stac_client_ecostress as stac_eco
from config import LOGGER, SITES_PILOTES, TIME_OF_INTEREST, OUTPUT_DIR, TIME_MARGIN_MINUTES, ltd, radius_km
from Utils import get_bbox_from_point

def main():
    LOGGER.info("🚀 DÉBUT DE LA PHASE 1 COUPLEE : EXTRACTION S2 + ECOSTRESS")

    # 1. Connexion aux deux catalogues
    catalog_s2 = stac_s2.connect_to_catalog()
    catalog_eco = stac_eco.connect_to_catalog()

    manifest_s2 = {}
    manifest_eco = {}

    stats_global = {"total_s2_trouvees": 0, "paires_trouvees": 0}

    # 2. Itération par site
    for pays, coords in SITES_PILOTES.items():
        LOGGER.info(f"\n========================================")
        LOGGER.info(f"🔍 Traitement du site : {pays}")

        bbox = get_bbox_from_point(coords["lon"], coords["lat"], radius_km)

        # Étape A: Récupération de TOUTES les images S2 de la période
        LOGGER.info("   -> Recherche globale Sentinel-2...")
        items_s2_bruts = stac_s2.search_images_s2(catalog_s2, bbox, TIME_OF_INTEREST, pays)
        
        # Filtrage local des nuages S2
        items_s2_propres = [item for item in items_s2_bruts if item.properties.get("eo:cloud_cover", 100) <= ltd]
        LOGGER.info(f"   -> {len(items_s2_propres)} images S2 retenues après filtre nuages (<= {ltd}%).")

        stats_global["total_s2_trouvees"] += len(items_s2_propres)

        paires_site_s2 = []
        paires_site_eco = []

        # Étape B: Pour chaque image S2, chercher une ECOSTRESS correspondante
        for item_s2 in items_s2_propres:
            s2_dt = item_s2.datetime
            
            # Définir la fenêtre de recherche autour de l'acquisition S2
            dt_start = s2_dt - timedelta(minutes=TIME_MARGIN_MINUTES)
            dt_end = s2_dt + timedelta(minutes=TIME_MARGIN_MINUTES)
            time_window = f"{dt_start.strftime('%Y-%m-%dT%H:%M:%SZ')}/{dt_end.strftime('%Y-%m-%dT%H:%M:%SZ')}"

            # Recherche ECOSTRESS sur cette fenêtre très réduite
            LOGGER.info(f"      Recherche ECOSTRESS pour S2 du {s2_dt.strftime('%Y-%m-%d %Hh%M')} (±{TIME_MARGIN_MINUTES}m)...")
            items_eco_match = stac_eco.search_images_ecostress(catalog_eco, bbox, time_window, pays)

            if items_eco_match:
                # Filtrer les images ECOSTRESS avec un angle de vue extrême (> 25°)
                items_eco_nadirs = []
                for it in items_eco_match:
                    # La clé exacte dans STAC NASA pour ECOSTRESS est souvent 'view:off_nadir'
                    off_nadir = it.properties.get("view:off_nadir")
                    # Parfois, il peut être absent ou None, on l'accepte s'il est <= 25 ou inconnu (0)
                    if off_nadir is None:
                        off_nadir = 0
                    if abs(float(off_nadir)) <= 25:
                        items_eco_nadirs.append(it)

                if items_eco_nadirs:
                    # Si plusieurs ECOSTRESS correspondent (rare sur 1h), on prend la plus proche temporellement
                    best_eco = min(items_eco_nadirs, key=lambda it: abs((it.datetime - s2_dt).total_seconds()))
                    delta = abs((best_eco.datetime - s2_dt).total_seconds()) / 60
                    
                    LOGGER.info(f"      ✅ PAIRE TROUVÉE ! Delta = {delta:.1f} min (Angle={best_eco.properties.get('view:off_nadir', 'N/A')}°).")
                    paires_site_s2.append(item_s2.id)
                    paires_site_eco.append(best_eco.id)
                    stats_global["paires_trouvees"] += 1
                else:
                    LOGGER.info("      ❌ ECOSTRESS trouvé mais rejeté (Angle de vue trop grand > 25°).")
            else:
                LOGGER.info("      ❌ Pas d'ECOSTRESS dans cette fenêtre.")

        # Dé-duplication des IDs (au cas où, bien que peu probable avec ce workflow)
        paires_site_s2 = list(dict.fromkeys(paires_site_s2))
        paires_site_eco = list(dict.fromkeys(paires_site_eco))

        manifest_s2[pays] = paires_site_s2
        manifest_eco[pays] = paires_site_eco

        LOGGER.info(f"   📊 Bilan {pays} : {len(paires_site_s2)} paires validées.")

    # 3. Sauvegarde des manifestes (ils vont remplacer les anciens)
    path_s2 = os.path.join(OUTPUT_DIR, "manifest_extraction_s2.json")
    path_eco = os.path.join(OUTPUT_DIR, "manifest_extraction_ecostress.json")

    with open(path_s2, "w", encoding="utf-8") as f:
        json.dump(manifest_s2, f, indent=4)

    with open(path_eco, "w", encoding="utf-8") as f:
        json.dump(manifest_eco, f, indent=4)

    LOGGER.info("\n✅ PHASE 1 COUPLEE TERMINÉE.")
    LOGGER.info(f"   📈 Stats Globales : {stats_global['paires_trouvees']} paires trouvées sur {stats_global['total_s2_trouvees']} images S2 sans nuages.")
    LOGGER.info(f"   ⚙️ Manifeste S2 écrasé avec les paires : {path_s2}")
    LOGGER.info(f"   ⚙️ Manifeste ECOSTRESS écrasé avec les paires : {path_eco}")


if __name__ == "__main__":
    main()
