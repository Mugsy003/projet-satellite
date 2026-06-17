#!/usr/bin/env python3
"""Force-delete TIFs not listed in low-cloud manifests.

Usage:
  python tools/force_delete_non_lowcloud.py [--apply] [--site SITE]

Default: dry-run (lists files that would be removed). Use --apply to actually delete.
"""
import argparse
import json
import logging
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import ltd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def sites_from_outputs(outputs_dir):
    # find manifest files pattern manifest_extraction_S3_<site>.json
    files = os.listdir(outputs_dir)
    sites = set()
    for f in files:
        m = re.match(r"manifest_extraction_S3_(.+?)\.json$", f)
        if m:
            sites.add(m.group(1))
    return sorted(sites)


def load_manifest(path):
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except Exception:
        return None


def regenerate_lowcloud_dates(manifest, max_cloud):
    dates = []
    for pair in manifest:
        # manifest entries can be dicts with 'date' or may be strings/ids — skip non-dicts
        if not isinstance(pair, dict):
            continue
        d = pair.get('date')
        therm = pair.get('thermique', {})
        opt = pair.get('optique', {})
        t_cc = None
        o_cc = None
        try:
            t_cc = therm.get('properties', {}).get('eo:cloud_cover')
        except Exception:
            t_cc = None
        try:
            o_cc = opt.get('properties', {}).get('eo:cloud_cover')
        except Exception:
            o_cc = None
        # keep pair only if both cloud_cover are not None and <= max_cloud
        if d is None:
            continue
        if t_cc is None or o_cc is None:
            continue
        try:
            if float(t_cc) <= max_cloud and float(o_cc) <= max_cloud:
                dates.append(d)
        except Exception:
            continue
    return sorted(dates)


def list_tif_dates(tif_dir):
    if not os.path.isdir(tif_dir):
        return set()
    dates = set()
    for f in os.listdir(tif_dir):
        m = re.match(r"(\d{4}-\d{2}-\d{2})_", f)
        if m:
            dates.add(m.group(1))
    return dates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--apply', action='store_true', help='Actually delete files')
    p.add_argument('--site', help='Restrict to one site')
    args = p.parse_args()

    outputs_dir = os.path.join(ROOT, 'Outputs')
    if not os.path.isdir(outputs_dir):
        logging.error('Outputs directory not found: %s', outputs_dir)
        return

    sites = sites_from_outputs(outputs_dir)
    if args.site:
        if args.site not in sites:
            logging.error('Site %s not found in Outputs manifests', args.site)
            return
        sites = [args.site]

    total_removed = 0
    for site in sites:
        manifest_path = os.path.join(outputs_dir, f'manifest_extraction_S3_{site}.json')
        lowjson = os.path.join(outputs_dir, f'manifest_extraction_S3_{site}_lowcloud_dates.json')
        manifest = load_manifest(manifest_path)
        if manifest is None:
            logging.warning('No manifest for %s, skipping', site)
            continue

        # load existing lowcloud dates if present
        if os.path.exists(lowjson):
            try:
                low_dates = set(json.load(open(lowjson, 'r', encoding='utf-8')))
            except Exception:
                low_dates = set()
        else:
            low_dates = set()

        if not low_dates:
            # regenerate using manifest properties and config.ltd
            regen = regenerate_lowcloud_dates(manifest, ltd)
            low_dates = set(regen)
            logging.info('Regenerated %d low-cloud dates for %s (threshold %s%%)', len(low_dates), site, ltd)

        tif_dir = os.path.join(ROOT, 'Outputs', f'Serie_Temporelle_{site}_S3', 'TIF_Data')
        tif_dates = list_tif_dates(tif_dir)
        to_remove_dates = sorted(d for d in tif_dates if d not in low_dates)

        if not to_remove_dates:
            logging.info('Site %s: %d dates present, 0 to remove', site, len(tif_dates))
            continue

        logging.info('Site %s: %d dates present, %d dates to remove', site, len(tif_dates), len(to_remove_dates))
        files_to_remove = []
        for d in to_remove_dates:
            for f in os.listdir(tif_dir):
                if f.startswith(d + '_'):
                    files_to_remove.append(os.path.join(tif_dir, f))

        for fpath in files_to_remove:
            if args.apply:
                try:
                    os.remove(fpath)
                    logging.info('Removed %s', fpath)
                    total_removed += 1
                except Exception as e:
                    logging.error('Failed to remove %s: %s', fpath, e)
            else:
                logging.info('Would remove %s', fpath)

        # if apply, update lowcloud json to reflect that only low_dates are kept
        if args.apply:
            try:
                with open(lowjson, 'w', encoding='utf-8') as fh:
                    json.dump(sorted(low_dates), fh, ensure_ascii=False, indent=2)
                logging.info('Updated low-cloud manifest for %s: %s', site, lowjson)
            except Exception as e:
                logging.error('Failed to write %s: %s', lowjson, e)

    logging.info('Done. Total files removed: %d', total_removed)


if __name__ == '__main__':
    main()
