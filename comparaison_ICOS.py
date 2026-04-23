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

        # Consolidation dynamique des capteurs
        ts_cols = sorted([c for c in dobj.colNames if c.startswith('TS_')]) # [cite: 43]
        ta_cols = sorted([c for c in dobj.colNames if c.startswith('TA_')])
        
        df_icos['TS_Consolide'] = df_icos[ts_cols].bfill(axis=1).iloc[:, 0] if ts_cols else np.nan
        df_icos['TA_Consolide'] = df_icos[ta_cols].bfill(axis=1).iloc[:, 0] if ta_cols else np.nan
        
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
            # Chemin LST
            path_lst = os.path.join(tif_folder, f)
            # Chemin NDVI (on remplace LST_Sharpened_DMS par NDVI dans le nom du fichier)
            path_ndvi = path_lst.replace("LST_Sharpened_DMS", "NDVI")
            path_B10 = path_lst.replace("LST_Sharpened_DMS", "Thermique_B10")
            rds_lst = rioxarray.open_rasterio(path_lst)
            transformer = Transformer.from_crs("EPSG:4326", rds_lst.rio.crs, always_xy=True)
            x_p, y_p = transformer.transform(coords["lon"], coords["lat"])
            
            # Valeur LST
            pixel_val = rds_lst.sel(x=x_p, y=y_p, method="nearest").values[0]
            lst_sat = pixel_val - 273.15 if pixel_val > 200 else pixel_val

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
            lst_sat = np.nan
            ndvi_sat = np.nan
            b10_sat = np.nan

        if pd.isna(lst_sat):
            continue

        # 2. Recherche ICOS avec Marge de Tolérance
        diff_temps = abs(df_icos['TIMESTAMP'] - target_dt)
        masque_tolerance = diff_temps <= pd.Timedelta(minutes=TIME_MARGIN_MINUTES)
        valeurs_proches = df_icos[masque_tolerance]

        if not valeurs_proches.empty:
            index_le_plus_proche = diff_temps[masque_tolerance].idxmin()
            ligne_match = df_icos.loc[index_le_plus_proche]
            
            ts_ground = ligne_match['TS_Consolide']
            ta_ground = ligne_match['TA_Consolide']
            heure_icos = ligne_match['TIMESTAMP']
            
            if pd.notna(ts_ground):
                biais = abs(lst_sat - ts_ground)
                decalage_min = (heure_icos - target_dt).total_seconds() / 60
                
                resultats_globaux.append({
                    "Site": site,
                    "Date_Satellite": target_dt.strftime("%Y-%m-%d %H:%M"),
                    "NDVI": round(ndvi_sat, 3) if pd.notna(ndvi_sat) else "N/A",
                    "Heure_ICOS_Retenue": heure_icos.strftime("%H:%M"),
                    "Decalage_Temporel (min)": round(decalage_min, 1),
                    "LST_Sat (°C)": round(lst_sat, 2),
                    "ICOS_Sol (°C)": round(ts_ground, 2),
                    "ICOS_Air (°C)": round(ta_ground, 2),
                    "B10s (°C)": round(b10_sat, 2),
                    "Biais (°C)": round(biais, 2)
                })
                print(f"      ✅ Match ICOS à {heure_icos.strftime('%H:%M')} | NDVI: {ndvi_sat:.2f} | Biais: {biais:.2f} °C")
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