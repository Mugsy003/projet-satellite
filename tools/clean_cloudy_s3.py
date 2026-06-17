#!/usr/bin/env python3
"""
tools/clean_cloudy_s3.py
Supprime les TIFs et indices Sentinel-3 dont la couverture nuageuse
exceed le seuil `ltd` défini dans `config.py`.

Usage:
  python tools/clean_cloudy_s3.py [--site SITE] [--apply] [--verbose]

Par défaut le script fait un dry-run (n'efface rien). Passer `--apply`
pour effectuer les suppressions.
"""
import os
import json
import argparse
import logging
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from config import ...` works when
# script is executed from tools/ or other locations.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import OUTPUT_DIR, SITES_PILOTES, ltd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)


def load_manifest(site):
    path = os.path.join(OUTPUT_DIR, f"manifest_extraction_S3_{site}.json")
    if not os.path.exists(path):
        return None, path
    with open(path, 'r') as f:
        return json.load(f), path


def save_manifest(site, pairs):
    path = os.path.join(OUTPUT_DIR, f"manifest_extraction_S3_{site}.json")
    with open(path, 'w') as f:
        json.dump(pairs, f, indent=2)


def save_lowcloud_dates(site, dates):
    path = os.path.join(OUTPUT_DIR, f"manifest_extraction_S3_{site}_lowcloud_dates.json")
    with open(path, 'w') as f:
        json.dump(dates, f, indent=2)


def get_cloud_cover_from_item(item):
    # item can be a dict (to_dict()) or an id string
    if not isinstance(item, dict):
        return None
    # common places: properties.eo:cloud_cover or top-level 'eo:cloud_cover' or 'cloud_cover'
    props = item.get('properties', {}) if isinstance(item.get('properties', {}), dict) else {}
    cc = props.get('eo:cloud_cover')
    if cc is None:
        cc = props.get('cloud_cover')
    if cc is None:
        cc = item.get('eo:cloud_cover')
    if cc is None:
        cc = item.get('cloud_cover')
    try:
        return float(cc) if cc is not None else None
    except Exception:
        return None


def clean_site(site, apply=False, verbose=False):
    manifest, path = load_manifest(site)
    if manifest is None:
        LOGGER.info(f"No manifest for site {site} at {path}, skipping.")
        return

    dossier_sortie = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}_S3", "TIF_Data")

    to_delete_dates = []
    retained_pairs = []

    for paire in manifest:
        date = paire.get('date')
        item_t = paire.get('thermique')
        item_o = paire.get('optique')

        cc_t = get_cloud_cover_from_item(item_t)
        cc_o = get_cloud_cover_from_item(item_o)

        too_cloudy = False
        if cc_t is not None and cc_t > ltd:
            too_cloudy = True
        if cc_o is not None and cc_o > ltd:
            too_cloudy = True

        if too_cloudy:
            to_delete_dates.append((date, cc_t, cc_o))
        else:
            retained_pairs.append(paire)

    LOGGER.info(f"Site {site}: {len(manifest)} paires, {len(to_delete_dates)} à supprimer (seuil {ltd}%).")

    # Delete files for dates to delete
    for date, cc_t, cc_o in to_delete_dates:
        prefix = f"{date}_{site}"
        if os.path.exists(dossier_sortie):
            for fname in os.listdir(dossier_sortie):
                if fname.startswith(prefix):
                    path_rm = os.path.join(dossier_sortie, fname)
                    LOGGER.info(f"Will remove: {path_rm}  (cc SLSTR={cc_t}, Synergy={cc_o})")
                    if apply:
                        try:
                            if os.path.isdir(path_rm):
                                shutil.rmtree(path_rm)
                            else:
                                os.remove(path_rm)
                            LOGGER.info(f"Removed: {path_rm}")
                        except Exception as e:
                            LOGGER.warning(f"Failed to remove {path_rm}: {e}")
        else:
            LOGGER.info(f"Output folder {dossier_sortie} does not exist, skipping file deletes for {date}.")

    # Update manifest and lowcloud dates JSON
    if len(retained_pairs) != len(manifest):
        if apply:
            save_manifest(site, retained_pairs)
            dates = [p.get('date') for p in retained_pairs]
            save_lowcloud_dates(site, dates)
            LOGGER.info(f"Updated manifest and lowcloud dates for {site}: {len(retained_pairs)} pairs remain.")
        else:
            LOGGER.info(f"Dry-run: would update manifest for {site}: {len(retained_pairs)} pairs remain.")
    else:
        LOGGER.info(f"No changes to manifest for {site}.")


def main():
    parser = argparse.ArgumentParser(description='Clean cloudy Sentinel-3 TIFs based on manifest properties')
    parser.add_argument('--site', help='Process single site name (as in config.SITES_PILOTES)')
    parser.add_argument('--apply', action='store_true', help='Actually delete files (default: dry-run)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)

    sites = [args.site] if args.site else list(SITES_PILOTES.keys())

    for site in sites:
        if site not in SITES_PILOTES:
            LOGGER.warning(f"Unknown site {site}, skipping.")
            continue
        clean_site(site, apply=args.apply, verbose=args.verbose)


if __name__ == '__main__':
    main()
