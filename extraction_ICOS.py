import os
import pandas as pd
import numpy as np
from icoscp.dobj import Dobj
from icoscp_core.icos import auth
from config import PIDS_ICOS


def main():
    print("🔐 Initialisation de l'authentification ICOS...")
    # auth.init_config_file() # À décommenter si tu ne l'as pas encore fait sur cette machine

    dossier_sortie = "Outputs_ICOS"
    os.makedirs(dossier_sortie, exist_ok=True)

    for site, pid in PIDS_ICOS.items():
        print(f"\n========================================")
        print(f"🌍 Traitement du site : {site}")
        
        # 1. Chargement de l'objet via son PID [cite: 19]
        try:
            dobj = Dobj(pid)
        except Exception as e:
            print(f"❌ Erreur lors de l'accès au PID {pid}: {e}")
            continue

        if not dobj.valid: # Vérification de la validité [cite: 150]
            print(f"❌ Objet invalide pour {site}.")
            continue

        print(f"✅ Objet chargé : {dobj.id}")

        # 2. Identification dynamique des colonnes 
        toutes_les_colonnes = dobj.colNames
        if not toutes_les_colonnes:
            print(f"⚠️ Aucune colonne trouvée pour {site}.")
            continue

        # On isole les colonnes de rayonnement (LW_IN et LW_OUT)
        lw_in_cols = sorted([col for col in toutes_les_colonnes if col.startswith('LW_IN_')])
        lw_out_cols = sorted([col for col in toutes_les_colonnes if col.startswith('LW_OUT_')])
        
        print(f"   🔍 Capteurs LW_IN trouvés : {lw_in_cols}")
        print(f"   🔍 Capteurs LW_OUT trouvés : {lw_out_cols}")

        colonnes_a_extraire = ['TIMESTAMP'] + lw_in_cols + lw_out_cols

        # 3. Extraction des données [cite: 170]
        df = dobj.get(columns=colonnes_a_extraire)

        # Traitement temporel
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df.set_index('TIMESTAMP', inplace=True)

        # Nettoyage des valeurs invalides ICOS [cite: 459]
        df.replace([-9.99, -999.0, -9999.0], np.nan, inplace=True)

        # 4. FUSION INTELLIGENTE DES CAPTEURS
        # On crée une colonne unique pour LW_IN et LW_OUT
        # .bfill(axis=1) prend la première valeur non-NaN trouvée de gauche à droite
        if lw_in_cols:
            df['LW_IN_Consolide'] = df[lw_in_cols].bfill(axis=1).iloc[:, 0]
        else:
            df['LW_IN_Consolide'] = np.nan

        if lw_out_cols:
            df['LW_OUT_Consolide'] = df[lw_out_cols].bfill(axis=1).iloc[:, 0]
        else:
            df['LW_OUT_Consolide'] = np.nan

        # Calcul de la température de surface (LST)
        sigma = 5.67e-8
        emissivite = 0.98  # Valeur pour Gebesee, ajustable par site si besoin

        df['LST_Calculee'] = ((df['LW_OUT_Consolide'] - (1 - emissivite) * df['LW_IN_Consolide']) / (emissivite * sigma))**0.25 - 273.15

        # 5. Filtrage pour l'heure Landsat (10h20 - 10h40)
        df_validation = df.between_time('10:20', '10:40')

        # On ne garde que les colonnes finales pour que le CSV soit propre
        colonnes_finales = ['LW_IN_Consolide', 'LW_OUT_Consolide', 'LST_Calculee']
        df_final = df_validation[colonnes_finales].dropna(how='all')

        if df_final.empty:
            print(f"   ⚠️ Aucune donnée valide trouvée entre 10h20 et 10h40 pour {site}.")
            continue

        print(f"   🏆 Aperçu des valeurs pour {site} :")
        print(df_final.tail(3))

        # 6. Sauvegarde
        chemin_csv = os.path.join(dossier_sortie, f"donnees_icos_{site}.csv")
        df_final.to_csv(chemin_csv)
        print(f"   💾 Fichier sauvegardé : {chemin_csv}")

if __name__ == "__main__":
    main()