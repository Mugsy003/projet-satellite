import os
import sys
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
import pandas as pd
import numpy as np
import pvlib
from config import PIDS_NOAA, TIME_OF_INTEREST


def main():
    print("🔐 Initialisation de l'extraction NOAA (SURFRAD)...")

    dossier_sortie = "Outputs_NOAA"
    os.makedirs(dossier_sortie, exist_ok=True)

    for site, pid in PIDS_NOAA.items():
        print(f"\n========================================")
        print(f"🌍 Traitement du site NOAA : {site} (code : {pid})")

        # 1. Détermination des dates à extraire (comme pour ICOS mais adapté au format journalier)
        path_dates = os.path.join("Outputs", "manifest_dates_existantes.json")
        dates_a_extraire = []
        if os.path.exists(path_dates):
            try:
                with open(path_dates, "r", encoding="utf-8") as f:
                    manifest_dates = json.load(f)
                    dates_a_extraire = manifest_dates.get(site, [])
            except Exception as e:
                print(f"⚠️ Erreur lors de la lecture du manifeste des dates : {e}")

        if not dates_a_extraire:
            # Fallback : dates de la période d'intérêt globale
            try:
                start_str, end_str = TIME_OF_INTEREST.split('/')
                dates_a_extraire = pd.date_range(start=start_str, end=end_str).strftime("%Y-%m-%d").tolist()
                print(f"   ℹ️ Pas de dates spécifiques trouvées dans le manifeste pour {site}.")
                print(f"      Utilisation de la période d'intérêt globale ({len(dates_a_extraire)} jours)...")
            except Exception as e:
                print(f"❌ Erreur lors de la génération de la période d'intérêt : {e}")
                continue
        else:
            print(f"   ℹ️ {len(dates_a_extraire)} dates trouvées dans le manifeste pour {site}.")

        # 2. Téléchargement et parsing des fichiers journaliers NOAA SURFRAD
        dfs_site = []
        print("   📥 Téléchargement des données sur le serveur de la NOAA...")
        
        # On limite le nombre de requêtes à afficher pour ne pas surcharger la console
        for idx, date_str in enumerate(dates_a_extraire):
            dt = pd.to_datetime(date_str)
            year = dt.year
            year_short = dt.strftime("%y")
            jday = dt.timetuple().tm_yday
            
            url_surfrad = f"https://gml.noaa.gov/aftp/data/radiation/surfrad/{pid}/{year}/{pid}{year_short}{jday:03}.dat"
            
            if idx < 5 or idx >= len(dates_a_extraire) - 5:
                print(f"      [{idx+1}/{len(dates_a_extraire)}] Chargement : {url_surfrad}")
            elif idx == 5:
                print("      ...")
                
            try:
                # pvlib télécharge et parse automatiquement le fichier .dat
                df_day, meta = pvlib.iotools.read_surfrad(url_surfrad)
                dfs_site.append(df_day)
            except Exception as e:
                # Fichier potentiellement absent ou erreur réseau temporaire
                pass

        if not dfs_site:
            print(f"❌ Aucune donnée NOAA n'a pu être récupérée pour le site {site}.")
            continue

        print(f"✅ Données téléchargées pour {len(dfs_site)} jours.")
        
        # Fusion de tous les jours
        df = pd.concat(dfs_site)
        
        # Traitement temporel
        df.index = pd.to_datetime(df.index)
        
        # Nettoyage des valeurs invalides NOAA (-9999.0, etc.)
        df.replace([-9.99, -999.0, -9999.0], np.nan, inplace=True)
        # Certains capteurs peuvent avoir des valeurs négatives aberrantes
        df[df < -900] = np.nan

        # 3. FUSION INTELLIGENTE ET IDENTIFICATION DES COLONNES
        # 'dw_ir' = Downwelling Infrared (équivalent LW_IN)
        # 'uw_ir' = Upwelling Infrared (équivalent LW_OUT)
        df['LW_IN_Consolide'] = df['dw_ir']
        df['LW_OUT_Consolide'] = df['uw_ir']

        # Calcul de la température de surface (LST)
        sigma = 5.67e-8
        emissivite = 0.98  # Valeur par défaut ajustable
        
        df['LST_Calculee'] = ((df['LW_OUT_Consolide'] - (1 - emissivite) * df['LW_IN_Consolide']) / (emissivite * sigma))**0.25 - 273.15

        # 4. Filtrage dynamique pour l'heure Landsat en temps universel (UTC)
        # Landsat passe à environ 10h30 heure solaire locale. On convertit en UTC :
        # UTC = Heure locale - (Longitude / 15)
        lon = 0
        from config import SITES_PILOTES
        if site in SITES_PILOTES:
            lon = SITES_PILOTES[site]["lon"]
            
        heure_passage_utc_dec = 10.5 - (lon / 15.0)
        heure_passage_utc_dec = heure_passage_utc_dec % 24
        
        heure = int(heure_passage_utc_dec)
        minute = int((heure_passage_utc_dec - heure) * 60)
        
        from datetime import datetime, timedelta
        pass_time = datetime(2000, 1, 1, heure, minute)
        # On garde une marge large de +/- 1 heure autour du passage satellite
        start_time = (pass_time - timedelta(minutes=60)).strftime('%H:%M')
        end_time = (pass_time + timedelta(minutes=60)).strftime('%H:%M')

        df_validation = df.between_time(start_time, end_time)

        # Colonnes finales propres
        colonnes_finales = ['LW_IN_Consolide', 'LW_OUT_Consolide', 'LST_Calculee']
        df_final = df_validation[colonnes_finales].dropna(how='all')

        if df_final.empty:
            print(f"   ⚠️ Aucune donnée valide trouvée entre {start_time} et {end_time} UTC pour {site}.")
            continue

        print(f"   🏆 Aperçu des valeurs pour {site} (filtré autour de {start_time}-{end_time} UTC) :")
        print(df_final.tail(3))

        # 5. Sauvegarde
        chemin_csv = os.path.join(dossier_sortie, f"donnees_noaa_{site}.csv")
        df_final.to_csv(chemin_csv)
        print(f"   💾 Fichier sauvegardé : {chemin_csv}")

if __name__ == "__main__":
    main()