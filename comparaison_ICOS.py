import os
import re
import rioxarray
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pyproj import Transformer
# pyrefly: ignore [missing-import]
from icoscp.dobj import Dobj 
from config import SITES_PILOTES, PIDS_ICOS, TIME_OF_INTEREST, TIME_MARGIN_MINUTES

# --- CONFIGURATION DES CHEMINS ET PARAMÈTRES ---
BASE_TIF_DIR = r"C:\Users\a951444\Workspace\projet-satellite\Outputs"

def extract_datetime_from_filename(filename):
    """
    Extrait la date et l'heure d'un nom de fichier TIF.
    Gère les formats: 'YYYY-MM-DD' et 'YYYY-MM-DD_HHhMM'
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})(?:_(\d{2})h(\d{2}))?", filename)
    if match:
        date_str = match.group(1)
        hour_str = match.group(2) if match.group(2) else "10"
        min_str = match.group(3) if match.group(3) else "30"
        return pd.to_datetime(f"{date_str} {hour_str}:{min_str}:00")
    return None

# ==========================================
# 1. PRÉPARATION DE LA PÉRIODE D'INTÉRÊT
# ==========================================
start_str, end_str = TIME_OF_INTEREST.split('/')
start_date = pd.to_datetime(start_str)
end_date = pd.to_datetime(end_str)

resultats_globaux = []

# ==========================================
# 2. BOUCLE SUR TOUS LES SITES
# ==========================================
for site, coords in SITES_PILOTES.items():
    print(f"\n" + "="*60)
    print(f"🌍 TRAITEMENT DU SITE : {site}")
    
    pid = PIDS_ICOS.get(site)
    df_icos = None

    # --- A. Chargement des données Deep Learning et ICOS ---
    dl_file = os.path.join("deep_learning2", f"stress_hydrique_{site.lower()}.csv")
    df_dl = None
    if os.path.exists(dl_file):
        df_dl = pd.read_csv(dl_file)
        df_dl['date'] = pd.to_datetime(df_dl['date']).dt.date

    # --- Chargement des données GOL locales ---
    site_map = {"Italy": "Italie", "Greece": "Grece"}
    site_suffix = site_map.get(site, site)
    gol_file = os.path.join("donnees_Gol", f"temperatures_{site_suffix}.csv")
    df_gol = None
    if os.path.exists(gol_file):
        df_gol = pd.read_csv(gol_file)
        # Gestion robuste des dates avec fuseaux horaires
        df_gol['created_date'] = pd.to_datetime(df_gol['created_date'], utc=True).dt.tz_localize(None)

    if not pid:
        print(f"⚠️ Aucun PID ICOS trouvé pour {site}. On traitera avec Satellite + Deep Learning uniquement.")
    else:
        print(f"🌡️ Chargement de la base de données ICOS (PID: {pid})...")
        try:
            dobj = Dobj(pid) # [cite: 5, 22]
            if not dobj.valid: # [cite: 148, 156]
                raise ValueError("Objet ICOS invalide.")
                
            df_icos = dobj.data # [cite: 51, 54]
            df_icos['TIMESTAMP'] = pd.to_datetime(df_icos['TIMESTAMP'])
            df_icos.replace([-9.99, -999.0, -9999.0], np.nan, inplace=True)

            # Consolidation dynamique des capteurs (LW_IN et LW_OUT)
            lw_in_cols = sorted([c for c in dobj.colNames if c.startswith('LW_IN_')])
            lw_out_cols = sorted([c for c in dobj.colNames if c.startswith('LW_OUT_')])
            
            df_icos['LW_IN_Consolide'] = df_icos[lw_in_cols].bfill(axis=1).iloc[:, 0] if lw_in_cols else np.nan
            df_icos['LW_OUT_Consolide'] = df_icos[lw_out_cols].bfill(axis=1).iloc[:, 0] if lw_out_cols else np.nan

            # Le calcul de LST_Calculee est différé dans la boucle TIF
            # afin d'utiliser une émissivité dynamique basée sur le NDVI du pixel.
            sigma = 5.67e-8
            
        except Exception as e:
            print(f"❌ Erreur ICOS pour {site} : {e}")

    # --- B. Recherche des fichiers TIF locaux ---
    tif_folder = os.path.join(BASE_TIF_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data")
    if not os.path.exists(tif_folder):
        print(f"❌ Dossier TIF introuvable : {tif_folder}")
        continue

    fichiers_tif = [f for f in os.listdir(tif_folder) if f.endswith('.tif') and "LST_Sharpened_DMS" in f]
    print(f"📂 {len(fichiers_tif)} fichiers TIF détectés. Analyse en cours...")

    # --- C. Comparaison Temporelle Intelligente ---
    for f in fichiers_tif:
        target_dt = extract_datetime_from_filename(f)
        if not target_dt: continue
        
        # On ne filtre plus par rapport à TIME_OF_INTEREST pour pouvoir analyser toutes les images téléchargées
        # if not (start_date <= target_dt <= end_date):
        #     continue

        print(f"   🔍 Analyse satellite du {target_dt}...")

        # Valeur Deep Learning
        lst_sat_dl = np.nan
        if df_dl is not None:
            match_dl = df_dl[df_dl['date'] == target_dt.date()]
            if not match_dl.empty:
                lst_sat_dl = match_dl.iloc[0]['temperature_C']

        # 1. Extraction de la Valeur Satellite (LST et NDVI)
        try:
            # Chemin LST DMS
            path_lst_dms = os.path.join(tif_folder, f)
            # Chemin LST TsHARP
            path_lst_tsharp = path_lst_dms.replace("LST_Sharpened_DMS", "LST_Sharpened_TsHARP")
            # Chemin LST DMS Fusion
            path_lst_dms_fusion = path_lst_dms.replace("LST_Sharpened_DMS", "LST_Sharpened_DMS_Fusion")
            # Chemin LST TsHARP Fusion
            path_lst_tsharp_fusion = path_lst_dms.replace("LST_Sharpened_DMS", "LST_Sharpened_TsHARP_Fusion")
            
            # Chemin NDVI et B10
            path_ndvi = path_lst_dms.replace("LST_Sharpened_DMS", "NDVI")
            path_B10 = path_lst_dms.replace("LST_Sharpened_DMS", "Thermique_B10")
            
            rds_lst_dms = rioxarray.open_rasterio(path_lst_dms)
            transformer = Transformer.from_crs("EPSG:4326", rds_lst_dms.rio.crs, always_xy=True)
            x_p, y_p = transformer.transform(coords["lon"], coords["lat"])
            
            # Valeur LST DMS
            pixel_val_dms = rds_lst_dms.sel(x=x_p, y=y_p, method="nearest").values[0]
            lst_sat_dms = pixel_val_dms - 273.15 if pixel_val_dms > 200 else pixel_val_dms

            # Valeur LST TsHARP
            if os.path.exists(path_lst_tsharp):
                rds_lst_tsharp = rioxarray.open_rasterio(path_lst_tsharp)
                pixel_val_tsharp = rds_lst_tsharp.sel(x=x_p, y=y_p, method="nearest").values[0]
                lst_sat_tsharp = pixel_val_tsharp - 273.15 if pixel_val_tsharp > 200 else pixel_val_tsharp
            else:
                lst_sat_tsharp = np.nan

            # Valeur LST DMS Fusion (10m - grille S2 differente, besoin d'un nouveau transformer)
            lst_sat_dms_fusion = np.nan
            if os.path.exists(path_lst_dms_fusion):
                rds_fusion = rioxarray.open_rasterio(path_lst_dms_fusion)
                tf_fusion = Transformer.from_crs("EPSG:4326", rds_fusion.rio.crs, always_xy=True)
                xf, yf = tf_fusion.transform(coords["lon"], coords["lat"])
                pv = rds_fusion.sel(x=xf, y=yf, method="nearest").values[0]
                lst_sat_dms_fusion = pv - 273.15 if pv > 200 else pv

            # Valeur LST TsHARP Fusion (10m)
            lst_sat_tsharp_fusion = np.nan
            if os.path.exists(path_lst_tsharp_fusion):
                rds_fusion_ts = rioxarray.open_rasterio(path_lst_tsharp_fusion)
                tf_fusion_ts = Transformer.from_crs("EPSG:4326", rds_fusion_ts.rio.crs, always_xy=True)
                xft, yft = tf_fusion_ts.transform(coords["lon"], coords["lat"])
                pvt = rds_fusion_ts.sel(x=xft, y=yft, method="nearest").values[0]
                lst_sat_tsharp_fusion = pvt - 273.15 if pvt > 200 else pvt

            # Valeur NDVI
            if os.path.exists(path_ndvi):
                rds_ndvi = rioxarray.open_rasterio(path_ndvi)
                ndvi_sat = rds_ndvi.sel(x=x_p, y=y_p, method="nearest").values[0]
            else:
                ndvi_sat = np.nan

            # --- Émissivité dynamique basée sur le NDVI ---
            # fraction_vegetation selon la méthode de Sobrino et al.
            NDVI_SOL = 0.2   # NDVI typique d'un sol nu
            NDVI_VEG = 0.86  # NDVI typique d'une végétation dense
            if pd.notna(ndvi_sat):
                fv = ((ndvi_sat - NDVI_SOL) / (NDVI_VEG - NDVI_SOL)) ** 2
                fraction_vegetation = float(np.clip(fv, 0.0, 1.0))
            else:
                fraction_vegetation = 0.5  # valeur par défaut si NDVI indisponible
            emissivite_dynamique = 0.9332 + 0.0585 * fraction_vegetation

            # Recalcul de LST_Calculee ICOS avec l'émissivité dynamique
            if df_icos is not None:
                df_icos['LST_Calculee'] = (
                    (df_icos['LW_OUT_Consolide'] - (1 - emissivite_dynamique) * df_icos['LW_IN_Consolide'])
                    / (emissivite_dynamique * sigma)
                ) ** 0.25 - 273.15
            # Valeur B10
            if os.path.exists(path_B10):
                rds_B10 = rioxarray.open_rasterio(path_B10)
                b10_sat = rds_B10.sel(x=x_p, y=y_p, method="nearest").values[0]
            else:
                b10_sat = np.nan

        except Exception:
            lst_sat_dms = np.nan
            lst_sat_tsharp = np.nan
            lst_sat_dms_fusion = np.nan
            lst_sat_tsharp_fusion = np.nan
            ndvi_sat = np.nan
            b10_sat = np.nan

        if pd.isna(lst_sat_dms) and pd.isna(lst_sat_tsharp) and pd.isna(lst_sat_dms_fusion) and pd.isna(lst_sat_tsharp_fusion):
            continue

        # 2. Recherche ICOS avec Marge de Tolérance
        lst_ground = np.nan
        lw_in_ground = np.nan
        lw_out_ground = np.nan
        heure_icos = target_dt
        decalage_min = 0.0

        if df_icos is not None:
            diff_temps = abs(df_icos['TIMESTAMP'] - target_dt)
            masque_tolerance = diff_temps <= pd.Timedelta(minutes=TIME_MARGIN_MINUTES)
            valeurs_proches = df_icos[masque_tolerance]

            if not valeurs_proches.empty:
                index_le_plus_proche = diff_temps[masque_tolerance].idxmin()
                ligne_match = df_icos.loc[index_le_plus_proche]
                
                lst_ground = ligne_match['LST_Calculee']
                lw_in_ground = ligne_match['LW_IN_Consolide']
                lw_out_ground = ligne_match['LW_OUT_Consolide']
                heure_icos = ligne_match['TIMESTAMP']
                decalage_min = (heure_icos - target_dt).total_seconds() / 60
            else:
                print(f"      ⚠️ Aucune donnée ICOS trouvée dans une marge de +/- {TIME_MARGIN_MINUTES} min.")

        # 3. Recherche GOL (Terrain) - On extrait TSOIL et TAIR separement
        # car Landsat mesure la temperature de SURFACE (skin temperature),
        # et ni Tsoil (enterre) ni Tair (2m au-dessus) n'est un proxy parfait.
        # On garde les deux pour comparer empiriquement.
        temp_gol_tsoil = np.nan
        temp_gol_tair = np.nan
        heure_gol = target_dt
        if df_gol is not None:
            diff_temps_gol = abs(df_gol['created_date'] - target_dt)
            masque_gol = diff_temps_gol <= pd.Timedelta(minutes=TIME_MARGIN_MINUTES)
            if not df_gol[masque_gol].empty:
                idx_gol = diff_temps_gol[masque_gol].idxmin()
                heure_gol = df_gol.loc[idx_gol, 'created_date']
                
                # Tair (air a 2m)
                if 'field_tair_c_avg' in df_gol.columns:
                    temp_gol_tair = df_gol.loc[idx_gol, 'field_tair_c_avg']
                
                # Tsoil (sonde la moins profonde disponible)
                tsoil_priority = ['field_tsoil_a_10_avg', 'field_tsoil_a_20_avg', 'field_tsoil_a_25_avg']
                for col_tsoil in tsoil_priority:
                    if col_tsoil in df_gol.columns:
                        val = df_gol.loc[idx_gol, col_tsoil]
                        if pd.notna(val) and val > -30:
                            temp_gol_tsoil = val
                            break
        
        # On utilise Tair comme valeur GOL par defaut (plus courant dans la litterature)
        temp_gol = temp_gol_tair if pd.notna(temp_gol_tair) else temp_gol_tsoil

        biais_dms = abs(lst_sat_dms - lst_ground) if pd.notna(lst_sat_dms) and pd.notna(lst_ground) else np.nan
        biais_tsharp = abs(lst_sat_tsharp - lst_ground) if pd.notna(lst_sat_tsharp) and pd.notna(lst_ground) else np.nan
        biais_dms_fusion = abs(lst_sat_dms_fusion - lst_ground) if pd.notna(lst_sat_dms_fusion) and pd.notna(lst_ground) else np.nan
        biais_tsharp_fusion = abs(lst_sat_tsharp_fusion - lst_ground) if pd.notna(lst_sat_tsharp_fusion) and pd.notna(lst_ground) else np.nan
        biais_dl = abs(lst_sat_dl - lst_ground) if pd.notna(lst_sat_dl) and pd.notna(lst_ground) else np.nan
        
        biais_dms_dl = abs(lst_sat_dms - lst_sat_dl) if pd.notna(lst_sat_dms) and pd.notna(lst_sat_dl) else np.nan
        biais_tsharp_dl = abs(lst_sat_tsharp - lst_sat_dl) if pd.notna(lst_sat_tsharp) and pd.notna(lst_sat_dl) else np.nan

        biais_dms_gol = abs(lst_sat_dms - temp_gol) if pd.notna(lst_sat_dms) and pd.notna(temp_gol) else np.nan
        biais_tsharp_gol = abs(lst_sat_tsharp - temp_gol) if pd.notna(lst_sat_tsharp) and pd.notna(temp_gol) else np.nan
        biais_dms_fusion_gol = abs(lst_sat_dms_fusion - temp_gol) if pd.notna(lst_sat_dms_fusion) and pd.notna(temp_gol) else np.nan
        biais_tsharp_fusion_gol = abs(lst_sat_tsharp_fusion - temp_gol) if pd.notna(lst_sat_tsharp_fusion) and pd.notna(temp_gol) else np.nan
        biais_dl_gol = abs(lst_sat_dl - temp_gol) if pd.notna(lst_sat_dl) and pd.notna(temp_gol) else np.nan

        resultats_globaux.append({
            "Site": site,
            "Date_Satellite": target_dt.strftime("%Y-%m-%d %H:%M"),
            "NDVI": round(ndvi_sat, 3) if pd.notna(ndvi_sat) else "N/A",
            "Heure_ICOS_Retenue": heure_icos.strftime("%H:%M") if pd.notna(lst_ground) else "N/A",
            "Heure_GOL_Retenue": heure_gol.strftime("%H:%M") if pd.notna(temp_gol) else "N/A",
            "Decalage_Temporel (min)": round(decalage_min, 1) if pd.notna(lst_ground) else "N/A",
            "LST_Sat_DMS (°C)": round(lst_sat_dms, 2) if pd.notna(lst_sat_dms) else "N/A",
            "LST_Sat_TsHARP (°C)": round(lst_sat_tsharp, 2) if pd.notna(lst_sat_tsharp) else "N/A",
            "LST_Sat_DMS_Fusion (°C)": round(lst_sat_dms_fusion, 2) if pd.notna(lst_sat_dms_fusion) else "N/A",
            "LST_Sat_TsHARP_Fusion (°C)": round(lst_sat_tsharp_fusion, 2) if pd.notna(lst_sat_tsharp_fusion) else "N/A",
            "LST_Sat_DL (°C)": round(lst_sat_dl, 2) if pd.notna(lst_sat_dl) else "N/A",
            "ICOS_LST (°C)": round(lst_ground, 2) if pd.notna(lst_ground) else "N/A",
            "Temperature_GOL (°C)": round(temp_gol, 2) if pd.notna(temp_gol) else "N/A",
            "Tsoil_GOL (°C)": round(temp_gol_tsoil, 2) if pd.notna(temp_gol_tsoil) else "N/A",
            "Tair_GOL (°C)": round(temp_gol_tair, 2) if pd.notna(temp_gol_tair) else "N/A",
            "B10s (°C)": round(b10_sat, 2) if pd.notna(b10_sat) else "N/A",
            "Biais_DMS_vs_ICOS (°C)": round(biais_dms, 2) if pd.notna(biais_dms) else "N/A",
            "Biais_TsHARP_vs_ICOS (°C)": round(biais_tsharp, 2) if pd.notna(biais_tsharp) else "N/A",
            "Biais_DMS_Fusion_vs_ICOS (°C)": round(biais_dms_fusion, 2) if pd.notna(biais_dms_fusion) else "N/A",
            "Biais_TsHARP_Fusion_vs_ICOS (°C)": round(biais_tsharp_fusion, 2) if pd.notna(biais_tsharp_fusion) else "N/A",
            "Biais_DL_vs_ICOS (°C)": round(biais_dl, 2) if pd.notna(biais_dl) else "N/A",
            "Biais_DMS_vs_GOL (°C)": round(biais_dms_gol, 2) if pd.notna(biais_dms_gol) else "N/A",
            "Biais_TsHARP_vs_GOL (°C)": round(biais_tsharp_gol, 2) if pd.notna(biais_tsharp_gol) else "N/A",
            "Biais_DMS_Fusion_vs_GOL (°C)": round(biais_dms_fusion_gol, 2) if pd.notna(biais_dms_fusion_gol) else "N/A",
            "Biais_TsHARP_Fusion_vs_GOL (°C)": round(biais_tsharp_fusion_gol, 2) if pd.notna(biais_tsharp_fusion_gol) else "N/A",
            "Biais_DL_vs_GOL (°C)": round(biais_dl_gol, 2) if pd.notna(biais_dl_gol) else "N/A",
        })
        
        if pd.notna(lst_ground):
            dl_str = f" | Biais DL: {biais_dl:.2f}degC" if pd.notna(biais_dl) else ""
            tsharp_str = f" | Biais TsHARP: {biais_tsharp:.2f}degC" if pd.notna(biais_tsharp) else ""
            fusion_str = f" | DMS-Fus: {biais_dms_fusion:.2f}degC" if pd.notna(biais_dms_fusion) else ""
            fusion_ts_str = f" | TsH-Fus: {biais_tsharp_fusion:.2f}degC" if pd.notna(biais_tsharp_fusion) else ""
            print(f"      Match a {heure_icos.strftime('%H:%M')} | Biais DMS: {biais_dms:.2f}degC{tsharp_str}{fusion_str}{fusion_ts_str}{dl_str}")
        else:
            dl_str = f" | Biais DMS vs DL: {biais_dms_dl:.2f}degC" if pd.notna(biais_dms_dl) else ""
            print(f"      Extraction Sat reussie (Pas d'ICOS){dl_str}")

# ==========================================
# 3. SYNTHÈSE ET SAUVEGARDE
# ==========================================
if resultats_globaux:
    df_final = pd.DataFrame(resultats_globaux)
    print("\n" + "#"*70)
    print(f"📈 SYNTHÈSE GLOBALE ({TIME_OF_INTEREST}) - Tolérance: {TIME_MARGIN_MINUTES} min")
    print("#"*70)
    print(df_final.sort_values(['Site', 'Date_Satellite']).to_markdown(index=False))
    
    chemin_csv = os.path.join(BASE_TIF_DIR, "Validation_Saisonniere_LST.csv")
    df_final.to_csv(chemin_csv, index=False)
    print(f"\n💾 Résultats sauvegardés dans : {chemin_csv}")
else:
    print(f"\n⚠️ Aucune coïncidence trouvée entre les TIFs et ICOS (Tolérance: {TIME_MARGIN_MINUTES} min).")


