import os
import re
import rioxarray
import pandas as pd
import numpy as np
from pyproj import Transformer
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
    if not pid:
        print(f"⚠️ Aucun PID trouvé pour {site}, passage au site suivant.")
        continue

    # --- A. Chargement des données ICOS ---
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

        # Calcul de la LST
        sigma = 5.67e-8
        emissivite = 0.98
        df_icos['LST_Calculee'] = ((df_icos['LW_OUT_Consolide'] - (1 - emissivite) * df_icos['LW_IN_Consolide']) / (emissivite * sigma))**0.25 - 273.15
        
    except Exception as e:
        print(f"❌ Erreur ICOS pour {site} : {e}")
        continue

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
        
        # Vérifier si la date est dans la période configurée
        if not (start_date <= target_dt <= end_date):
            continue

        print(f"   🔍 Analyse satellite du {target_dt}...")

        # 1. Extraction de la Valeur Satellite (LST et NDVI)
        try:
            # Chemin LST DMS
            path_lst_dms = os.path.join(tif_folder, f)
            # Chemin LST TsHARP
            path_lst_tsharp = path_lst_dms.replace("LST_Sharpened_DMS", "LST_Sharpened_TsHARP")
            
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

            # Valeur NDVI
            if os.path.exists(path_ndvi):
                rds_ndvi = rioxarray.open_rasterio(path_ndvi)
                ndvi_sat = rds_ndvi.sel(x=x_p, y=y_p, method="nearest").values[0]
            else:
                ndvi_sat = np.nan
            # Valeur B10
            if os.path.exists(path_B10):
                rds_B10 = rioxarray.open_rasterio(path_B10)
                b10_sat = rds_B10.sel(x=x_p, y=y_p, method="nearest").values[0]
            else:
                b10_sat = np.nan

        except Exception:
            lst_sat_dms = np.nan
            lst_sat_tsharp = np.nan
            ndvi_sat = np.nan
            b10_sat = np.nan

        if pd.isna(lst_sat_dms) and pd.isna(lst_sat_tsharp):
            continue

        # 2. Recherche ICOS avec Marge de Tolérance
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
            
            if pd.notna(lst_ground):
                biais_dms = abs(lst_sat_dms - lst_ground) if pd.notna(lst_sat_dms) else np.nan
                biais_tsharp = abs(lst_sat_tsharp - lst_ground) if pd.notna(lst_sat_tsharp) else np.nan
                decalage_min = (heure_icos - target_dt).total_seconds() / 60
                
                resultats_globaux.append({
                    "Site": site,
                    "Date_Satellite": target_dt.strftime("%Y-%m-%d %H:%M"),
                    "NDVI": round(ndvi_sat, 3) if pd.notna(ndvi_sat) else "N/A",
                    "Heure_ICOS_Retenue": heure_icos.strftime("%H:%M"),
                    "Decalage_Temporel (min)": round(decalage_min, 1),
                    "LST_Sat_DMS (°C)": round(lst_sat_dms, 2) if pd.notna(lst_sat_dms) else "N/A",
                    "LST_Sat_TsHARP (°C)": round(lst_sat_tsharp, 2) if pd.notna(lst_sat_tsharp) else "N/A",
                    "ICOS_LST (°C)": round(lst_ground, 2),
                    "ICOS_LW_IN": round(lw_in_ground, 2),
                    "ICOS_LW_OUT": round(lw_out_ground, 2),
                    "B10s (°C)": round(b10_sat, 2),
                    "Biais_DMS (°C)": round(biais_dms, 2) if pd.notna(biais_dms) else "N/A",
                    "Biais_TsHARP (°C)": round(biais_tsharp, 2) if pd.notna(biais_tsharp) else "N/A"
                })
                print(f"      ✅ Match ICOS à {heure_icos.strftime('%H:%M')} | Biais DMS: {biais_dms:.2f}°C | Biais TsHARP: {biais_tsharp:.2f}°C" if pd.notna(biais_tsharp) else f"      ✅ Match ICOS à {heure_icos.strftime('%H:%M')} | Biais DMS: {biais_dms:.2f}°C | Biais TsHARP: N/A")
        else:
            print(f"      ⚠️ Aucune donnée ICOS trouvée dans une marge de +/- {TIME_MARGIN_MINUTES} min.")

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