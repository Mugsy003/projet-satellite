import os
import re
import json
import sys
import rioxarray
import pandas as pd
import numpy as np
from pyproj import Transformer
# pyrefly: ignore [missing-import]
from icoscp.dobj import Dobj
from config import SITES_PILOTES, PIDS_ICOS, PIDS_NOAA, TIME_OF_INTEREST, TIME_MARGIN_MINUTES, OUTPUT_DIR

sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION DES CHEMINS ---
BASE_TIF_DIR = OUTPUT_DIR
CSV_OUT_PATH = os.path.join(BASE_TIF_DIR, "Validation_Saisonniere_LST_S3.csv")

def extract_date_from_filename(filename):
    """
    Extrait la date YYYY-MM-DD d'un nom de fichier TIF Sentinel-3.
    Exemple: '2023-06-15_Lamasquere_LST_S3_1km.tif' -> '2023-06-15'
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return match.group(1)
    return None

def get_s3_datetime_mapping(site):
    """
    Parcourt les manifestes d'extraction S3 et de paires S3/S2 pour associer
    chaque date (YYYY-MM-DD) à l'heure d'overpass précise du satellite (en UTC).
    """
    mapping = {}
    
    # 1. Lecture du manifeste d'extraction S3
    path_s3 = os.path.join(OUTPUT_DIR, f"manifest_extraction_S3_{site}.json")
    if os.path.exists(path_s3):
        try:
            with open(path_s3, "r", encoding="utf-8") as f:
                paires = json.load(f)
                for p in paires:
                    date = p.get("date")
                    dt_str = p.get("thermique", {}).get("properties", {}).get("datetime")
                    if date and dt_str:
                        mapping[date] = pd.to_datetime(dt_str).tz_localize(None)
        except Exception as e:
            print(f"   [Debug] Erreur lecture manifeste S3 pour {site}: {e}")

    # 2. Lecture du manifeste de paires S3/S2 (complément/fusion)
    path_paires = os.path.join(OUTPUT_DIR, f"manifest_paires_S3_S2_{site}.json")
    if os.path.exists(path_paires):
        try:
            with open(path_paires, "r", encoding="utf-8") as f:
                paires = json.load(f)
                for p in paires:
                    date = p.get("date")
                    dt_str = p.get("s3", {}).get("properties", {}).get("datetime")
                    if date and dt_str:
                        mapping[date] = pd.to_datetime(dt_str).tz_localize(None)
        except Exception as e:
            print(f"   [Debug] Erreur lecture manifeste paires S3/S2 pour {site}: {e}")
            
    return mapping

# ==========================================
# 1. CONFIGURATION DE LA PÉRIODE
# ==========================================
start_str, end_str = TIME_OF_INTEREST.split('/')
start_date = pd.to_datetime(start_str)
end_date = pd.to_datetime(end_str)

resultats_globaux = []

# ==========================================
# 2. BOUCLE SUR TOUS LES SITES PILOTES
# ==========================================
for site, coords in SITES_PILOTES.items():
    print("\n" + "="*60)
    print(f"🌍 TRAITEMENT SENTINEL-3 SUR LE SITE : {site}")
    print("="*60)
    
    # --- A. Chargement des données ICOS / NOAA / GOL ---
    df_icos = None
    pid = PIDS_ICOS.get(site)
    
    # Chargement NOAA SURFRAD si applicable
    if site in PIDS_NOAA:
        print(f"🌡️  Chargement de la base de données NOAA pour {site}...")
        noaa_file = os.path.join("Outputs_NOAA", f"donnees_noaa_{site}.csv")
        if os.path.exists(noaa_file):
            df_icos = pd.read_csv(noaa_file)
            df_icos.rename(columns={df_icos.columns[0]: 'TIMESTAMP'}, inplace=True)
            df_icos['TIMESTAMP'] = pd.to_datetime(df_icos['TIMESTAMP']).dt.tz_localize(None)
        else:
            print(f"❌ Fichier NOAA introuvable : {noaa_file}")

    # Chargement ICOS (Local puis Fallback en ligne)
    elif pid:
        local_csv = os.path.join("Outputs_ICOS", f"donnees_icos_{site}.csv")
        if os.path.exists(local_csv):
            print(f"🌡️  Chargement des données ICOS locales depuis : {local_csv}...")
            try:
                df_icos = pd.read_csv(local_csv)
                df_icos['TIMESTAMP'] = pd.to_datetime(df_icos['TIMESTAMP'])
                df_icos.replace([-9.99, -999.0, -9999.0], np.nan, inplace=True)
            except Exception as e:
                print(f"❌ Erreur lors de la lecture du fichier ICOS local pour {site} : {e}")
                df_icos = None
        
        # Fallback en ligne si le fichier local n'existe pas ou a échoué
        if df_icos is None:
            print(f"🌡️  Connexion en ligne ICOS (PID: {pid})...")
            try:
                dobj = Dobj(pid)
                if not dobj.valid:
                    raise ValueError("Objet ICOS invalide.")
                df_icos = dobj.data
                df_icos['TIMESTAMP'] = pd.to_datetime(df_icos['TIMESTAMP'])
                df_icos.replace([-9.99, -999.0, -9999.0], np.nan, inplace=True)

                # Consolidation dynamique des capteurs (LW_IN et LW_OUT)
                lw_in_cols = sorted([c for c in dobj.colNames if c.startswith('LW_IN_') or c == 'LW_IN'])
                lw_out_cols = sorted([c for c in dobj.colNames if c.startswith('LW_OUT_') or c == 'LW_OUT'])
                
                df_icos['LW_IN_Consolide'] = df_icos[lw_in_cols].bfill(axis=1).iloc[:, 0] if lw_in_cols else np.nan
                df_icos['LW_OUT_Consolide'] = df_icos[lw_out_cols].bfill(axis=1).iloc[:, 0] if lw_out_cols else np.nan
            except Exception as e:
                print(f"❌ Erreur ICOS en ligne pour {site} : {e}")

    # Chargement des données GOL locales
    site_map = {"Italy": "Italie", "Greece": "Grece"}
    site_suffix = site_map.get(site, site)
    gol_file = os.path.join("donnees_Gol", f"temperatures_{site_suffix}.csv")
    df_gol = None
    if os.path.exists(gol_file):
        df_gol = pd.read_csv(gol_file)
        df_gol['created_date'] = pd.to_datetime(df_gol['created_date'], utc=True).dt.tz_localize(None)

    # --- B. Recherche des fichiers TIF locaux Sentinel-3 ---
    s3_folder = os.path.join(BASE_TIF_DIR, f"Serie_Temporelle_{site}_S3", "TIF_Data")
    if not os.path.exists(s3_folder):
        print(f"⚠️  Dossier Sentinel-3 introuvable pour {site} : {s3_folder}")
        continue

    # Récupérer les overpass exacts
    overpass_mapping = get_s3_datetime_mapping(site)
    
    dict_dates_paths = {} # date_str -> {'lst_1km': path, 'lst_300m': path, 'lst_fusion_10m': path, 'ndvi': path}
    
    for f in os.listdir(s3_folder):
        if not f.endswith('.tif'):
            continue
        date_str = extract_date_from_filename(f)
        if not date_str:
            continue
            
        if date_str not in dict_dates_paths:
            dict_dates_paths[date_str] = {}
            
        full_path = os.path.join(s3_folder, f)
        if "LST_S3_1km" in f:
            dict_dates_paths[date_str]['lst_1km'] = full_path
        elif "LST_S3_Sharpened_DMS_300m" in f:
            dict_dates_paths[date_str]['lst_300m'] = full_path
        elif "LST_Fusion_S3_S2_10m" in f:
            dict_dates_paths[date_str]['lst_fusion_10m'] = full_path
        elif "S3_NDVI" in f:
            dict_dates_paths[date_str]['ndvi'] = full_path

    if not dict_dates_paths:
        print(f"❌ Aucun fichier TIF Sentinel-3 trouvé dans {s3_folder}")
        continue

    # --- C. Filtrage par couverture nuageuse ---
    # On élimine les images contenant plus de 30% de valeurs de LST invalides (nuages)
    SEUIL_NUAGES_COMPARAISON = 30  # %
    dates_a_exclure = set()
    for date_str, paths in dict_dates_paths.items():
        lst_1km_path = paths.get('lst_1km')
        if lst_1km_path and os.path.exists(lst_1km_path):
            try:
                _ds = rioxarray.open_rasterio(lst_1km_path)
                _data = _ds.values.squeeze()
                _ds.close()
                _total = _data.size
                # On considère invalide/nuageux si nan ou hors de la plage normale sol [-30°C, 70°C]
                _valides = np.count_nonzero(np.isfinite(_data) & (_data > -30) & (_data < 70))
                _pct_nuages = (1 - _valides / _total) * 100
                if _pct_nuages > SEUIL_NUAGES_COMPARAISON:
                    dates_a_exclure.add(date_str)
                    print(f"   🚫 Date {date_str} exclue : trop nuageuse ({_pct_nuages:.1f}% de pixels invalides)")
            except Exception as e:
                print(f"   ⚠️ Impossible d'analyser les nuages pour la date {date_str}: {e}")

    for date_str in dates_a_exclure:
        del dict_dates_paths[date_str]

    if not dict_dates_paths:
        print(f"⚠️  Plus aucune date Sentinel-3 exploitable après filtrage nuageux.")
        continue

    print(f"📂 {len(dict_dates_paths)} dates Sentinel-3 valides prêtes pour la comparaison.")

    # --- D. Comparaison des données aux stations ---
    sigma = 5.67e-8

    for date_str, paths in sorted(dict_dates_paths.items()):
        # Déterminer la date et l'heure exactes de l'acquisition
        target_dt = overpass_mapping.get(date_str, pd.to_datetime(f"{date_str} 10:30:00"))
        print(f"   🔍 Analyse satellite du {target_dt.strftime('%Y-%m-%d %H:%M:%S')} (UTC)...")

        # 1. Extraction des valeurs satellites aux coordonnées de la station
        lst_1km_val = np.nan
        lst_300m_val = np.nan
        lst_fusion_val = np.nan
        ndvi_val = np.nan

        try:
            # LST 1km
            if paths.get('lst_1km') and os.path.exists(paths['lst_1km']):
                rds = rioxarray.open_rasterio(paths['lst_1km'])
                tf = Transformer.from_crs("EPSG:4326", rds.rio.crs, always_xy=True)
                x_p, y_p = tf.transform(coords["lon"], coords["lat"])
                v = rds.sel(x=x_p, y=y_p, method="nearest").values[0]
                lst_1km_val = v - 273.15 if v > 200 else v
                rds.close()

            # LST 300m (DMS)
            if paths.get('lst_300m') and os.path.exists(paths['lst_300m']):
                rds = rioxarray.open_rasterio(paths['lst_300m'])
                tf = Transformer.from_crs("EPSG:4326", rds.rio.crs, always_xy=True)
                x_p, y_p = tf.transform(coords["lon"], coords["lat"])
                v = rds.sel(x=x_p, y=y_p, method="nearest").values[0]
                lst_300m_val = v - 273.15 if v > 200 else v
                rds.close()

            # LST Fusion 10m (S3+S2)
            if paths.get('lst_fusion_10m') and os.path.exists(paths['lst_fusion_10m']):
                rds = rioxarray.open_rasterio(paths['lst_fusion_10m'])
                tf = Transformer.from_crs("EPSG:4326", rds.rio.crs, always_xy=True)
                x_p, y_p = tf.transform(coords["lon"], coords["lat"])
                v = rds.sel(x=x_p, y=y_p, method="nearest").values[0]
                lst_fusion_val = v - 273.15 if v > 200 else v
                rds.close()

            # NDVI
            if paths.get('ndvi') and os.path.exists(paths['ndvi']):
                rds = rioxarray.open_rasterio(paths['ndvi'])
                tf = Transformer.from_crs("EPSG:4326", rds.rio.crs, always_xy=True)
                x_p, y_p = tf.transform(coords["lon"], coords["lat"])
                ndvi_val = rds.sel(x=x_p, y=y_p, method="nearest").values[0]
                rds.close()
        except Exception as e:
            print(f"      ⚠️  Erreur extraction pixel pour {date_str} : {e}")

        # Si aucune mesure satellite n'est valide pour cette date, passer
        if pd.isna(lst_1km_val) and pd.isna(lst_300m_val) and pd.isna(lst_fusion_val):
            continue

        # 2. Calcul de l'émissivité dynamique et recalcul de la LST ICOS
        emissivite_dynamique = 0.98  # par défaut
        fraction_vegetation = np.nan
        if pd.notna(ndvi_val):
            NDVI_SOL = 0.2
            NDVI_VEG = 0.86
            fv = ((ndvi_val - NDVI_SOL) / (NDVI_VEG - NDVI_SOL)) ** 2
            fraction_vegetation = float(np.clip(fv, 0.0, 1.0))
            emissivite_dynamique = 0.9332 + 0.0585 * fraction_vegetation

        lst_ground = np.nan
        heure_icos = target_dt
        decalage_min = 0.0

        if df_icos is not None:
            # Calcul de LST avec émissivité dynamique spécifique à cette date
            df_icos['LST_Calculee_Dyn'] = (
                (df_icos['LW_OUT_Consolide'] - (1 - emissivite_dynamique) * df_icos['LW_IN_Consolide'])
                / (emissivite_dynamique * sigma)
            ) ** 0.25 - 273.15

            # Recherche de la ligne la plus proche en temps (+/- TIME_MARGIN_MINUTES)
            diff_temps = abs(df_icos['TIMESTAMP'] - target_dt)
            mask_margin = diff_temps <= pd.Timedelta(minutes=TIME_MARGIN_MINUTES)
            valeurs_proches = df_icos[mask_margin]

            if not valeurs_proches.empty:
                idx_proche = diff_temps[mask_margin].idxmin()
                ligne_match = df_icos.loc[idx_proche]
                lst_ground = ligne_match['LST_Calculee_Dyn']
                heure_icos = ligne_match['TIMESTAMP']
                decalage_min = (heure_icos - target_dt).total_seconds() / 60
            else:
                print(f"      ⚠️  Aucune donnée ICOS/NOAA dans la fenêtre de +/-{TIME_MARGIN_MINUTES} min.")

        # 3. Extraction GOL
        temp_gol = np.nan
        heure_gol = target_dt
        if df_gol is not None:
            diff_gol = abs(df_gol['created_date'] - target_dt)
            mask_gol = diff_gol <= pd.Timedelta(minutes=TIME_MARGIN_MINUTES)
            if not df_gol[mask_gol].empty:
                idx_gol = diff_gol[mask_gol].idxmin()
                heure_gol = df_gol.loc[idx_gol, 'created_date']
                
                # Priorité température de l'air ou sol superficiel
                if 'field_tair_c_avg' in df_gol.columns:
                    temp_gol = df_gol.loc[idx_gol, 'field_tair_c_avg']
                elif 'field_tsoil_a_10_avg' in df_gol.columns:
                    temp_gol = df_gol.loc[idx_gol, 'field_tsoil_a_10_avg']

        # 4. Calcul des écarts/biais
        bias_1km = lst_1km_val - lst_ground if pd.notna(lst_ground) else np.nan
        bias_300m = lst_300m_val - lst_ground if pd.notna(lst_ground) else np.nan
        bias_fusion = lst_fusion_val - lst_ground if pd.notna(lst_ground) else np.nan

        resultats_globaux.append({
            "Site": site,
            "Date_Satellite": target_dt.strftime("%Y-%m-%d %H:%M"),
            "NDVI_S3": round(ndvi_val, 3) if pd.notna(ndvi_val) else np.nan,
            "Fv": round(fraction_vegetation, 3) if pd.notna(fraction_vegetation) else np.nan,
            "Emis_Dyn": round(emissivite_dynamique, 4),
            "Heure_ICOS": heure_icos.strftime("%H:%M") if pd.notna(lst_ground) else "N/A",
            "Decalage_ICOS_min": round(decalage_min, 1) if pd.notna(lst_ground) else np.nan,
            "LST_S3_1km (°C)": round(lst_1km_val, 2) if pd.notna(lst_1km_val) else np.nan,
            "LST_S3_DMS_300m (°C)": round(lst_300m_val, 2) if pd.notna(lst_300m_val) else np.nan,
            "LST_S3_Fusion_10m (°C)": round(lst_fusion_val, 2) if pd.notna(lst_fusion_val) else np.nan,
            "ICOS_LST (°C)": round(lst_ground, 2) if pd.notna(lst_ground) else np.nan,
            "Temp_GOL (°C)": round(temp_gol, 2) if pd.notna(temp_gol) else np.nan,
            "Biais_1km_vs_ICOS (°C)": round(bias_1km, 2) if pd.notna(bias_1km) else np.nan,
            "Biais_300m_vs_ICOS (°C)": round(bias_300m, 2) if pd.notna(bias_300m) else np.nan,
            "Biais_Fusion_vs_ICOS (°C)": round(bias_fusion, 2) if pd.notna(bias_fusion) else np.nan
        })

        if pd.notna(lst_ground):
            s_dms = f" | Biais DMS 300m: {bias_300m:+.2f}°C" if pd.notna(bias_300m) else ""
            s_fus = f" | Biais Fusion 10m: {bias_fusion:+.2f}°C" if pd.notna(bias_fusion) else ""
            print(f"      Match sol à {heure_icos.strftime('%H:%M')} | Biais 1km: {bias_1km:+.2f}°C{s_dms}{s_fus}")
        else:
            print(f"      Sat OK: 1km={lst_1km_val:.1f}°C, 300m={lst_300m_val:.1f}°C (Pas de données sol)")

# ==========================================
# 3. STATISTIQUES FINALES & SAUVEGARDE
# ==========================================
if resultats_globaux:
    df_final = pd.DataFrame(resultats_globaux)
    
    print("\n" + "#"*75)
    print(f"📈 SYNTHÈSE DE VALIDATION SENTINEL-3 (Période: {TIME_OF_INTEREST})")
    print("#"*75)
    print(df_final.sort_values(['Site', 'Date_Satellite']).to_markdown(index=False))
    
    df_final.to_csv(CSV_OUT_PATH, index=False)
    print(f"\n💾 Résultats sauvegardés dans : {CSV_OUT_PATH}")
    
    # Calcul des métriques globales
    for col_sat, label in [("LST_S3_1km (°C)", "Raw 1km"), 
                           ("LST_S3_DMS_300m (°C)", "Sharpened DMS 300m"), 
                           ("LST_S3_Fusion_10m (°C)", "Fusion S3+S2 10m")]:
        mask_metrics = pd.notna(df_final[col_sat]) & pd.notna(df_final["ICOS_LST (°C)"])
        sub = df_final[mask_metrics]
        if not sub.empty:
            diff = sub[col_sat] - sub["ICOS_LST (°C)"]
            rmse = np.sqrt(np.mean(diff ** 2))
            mae = np.mean(np.abs(diff))
            bias = np.mean(diff)
            print(f"   🎯 Métriques globales [{label}] (N={len(sub)}) : RMSE = {rmse:.3f}°C | MAE = {mae:.3f}°C | Biais = {bias:+.3f}°C")
else:
    print(f"\n⚠️  Aucune coïncidence trouvée entre les TIFs Sentinel-3 et les stations sol.")
