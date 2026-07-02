import os
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
from icoscp.dobj import Dobj
# pyrefly: ignore [missing-import]
from icoscp_core.icos import auth
from config import PIDS_ICOS


# =============================================================================
# Variables météorologiques à extraire en plus du rayonnement LW
# Pour le modèle TTME (Two-source Trapezoid Model for Evapotranspiration)
# =============================================================================
PREFIXES_METEO = {
    "TA":    {"description": "Température de l'air (°C)",          "unite": "°C"},
    "WS":    {"description": "Vitesse du vent (m/s)",              "unite": "m/s"},
    "RH":    {"description": "Humidité relative (%)",              "unite": "%"},
    "SW_IN": {"description": "Rayonnement shortwave entrant (W/m²)", "unite": "W/m²"},
    "SW_OUT":{"description": "Rayonnement shortwave sortant (W/m²)", "unite": "W/m²"},
    "G":     {"description": "Flux de chaleur dans le sol (W/m²)",   "unite": "W/m²"},
    "PA":    {"description": "Pression atmosphérique (kPa)",       "unite": "kPa"},
    "VPD":   {"description": "Déficit de pression de vapeur (hPa)", "unite": "hPa"},
    "P":     {"description": "Précipitations (mm)",                "unite": "mm"},
}


def _find_columns_for_prefix(all_columns, prefix):
    """
    Cherche toutes les colonnes ICOS correspondant à un préfixe donné.
    Ex: prefix='TA' trouvera 'TA_F', 'TA_F_MDS', 'TA_1_1_1', etc.
    On priorise les colonnes gap-filled (_F) puis les colonnes brutes.
    """
    # Correspondance exacte ou avec suffixe commençant par _ 
    candidates = sorted([c for c in all_columns if c == prefix or c.startswith(prefix + '_')])
    return candidates


def _consolidate_columns(df, cols, name):
    """
    Fusionne plusieurs colonnes capteurs en une seule colonne consolidée.
    Priorité : colonnes gap-filled (_F) > colonnes brutes.
    """
    if not cols:
        return pd.Series(np.nan, index=df.index, name=name)
    
    # Prioriser les colonnes gap-filled
    priority = []
    others = []
    for c in cols:
        if '_F' in c:
            priority.append(c)
        else:
            others.append(c)
    ordered = priority + others
    
    return df[ordered].bfill(axis=1).iloc[:, 0].rename(name)


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

        # 2. Identification dynamique de TOUTES les colonnes disponibles
        toutes_les_colonnes = dobj.colNames
        if not toutes_les_colonnes:
            print(f"⚠️ Aucune colonne trouvée pour {site}.")
            continue

        print(f"   📋 Colonnes disponibles ({len(toutes_les_colonnes)}) : {toutes_les_colonnes}")

        # --- Colonnes de rayonnement LW (existant) ---
        lw_in_cols = sorted([col for col in toutes_les_colonnes if col.startswith('LW_IN_') or col == 'LW_IN'])
        lw_out_cols = sorted([col for col in toutes_les_colonnes if col.startswith('LW_OUT_') or col == 'LW_OUT'])
        
        print(f"   🔍 Capteurs LW_IN trouvés : {lw_in_cols}")
        print(f"   🔍 Capteurs LW_OUT trouvés : {lw_out_cols}")

        # --- Colonnes météo supplémentaires (nouveau pour TTME) ---
        meteo_cols_map = {}  # prefix -> list of column names
        for prefix, info in PREFIXES_METEO.items():
            cols = _find_columns_for_prefix(toutes_les_colonnes, prefix)
            if cols:
                meteo_cols_map[prefix] = cols
                print(f"   🌡️ {info['description']} : {cols}")
            else:
                print(f"   ⚠️ {info['description']} : NON DISPONIBLE")

        # 3. Extraction de TOUTES les données
        # On récupère tout le DataFrame via .data pour éviter les problèmes d'authentification
        try:
            df = dobj.data
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction des données pour {site}: {e}")
            continue

        # Traitement temporel
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df.set_index('TIMESTAMP', inplace=True)

        # Nettoyage des valeurs invalides ICOS [cite: 459]
        df.replace([-9.99, -999.0, -9999.0], np.nan, inplace=True)

        # 4. FUSION INTELLIGENTE DES CAPTEURS

        # --- Rayonnement LW (existant) ---
        if lw_in_cols:
            df['LW_IN_Consolide'] = df[lw_in_cols].bfill(axis=1).iloc[:, 0]
        else:
            df['LW_IN_Consolide'] = np.nan

        if lw_out_cols:
            df['LW_OUT_Consolide'] = df[lw_out_cols].bfill(axis=1).iloc[:, 0]
        else:
            df['LW_OUT_Consolide'] = np.nan

        # --- Variables météo (nouveau) ---
        for prefix, cols in meteo_cols_map.items():
            col_name = f"{prefix}_Consolide"
            df[col_name] = _consolidate_columns(df, cols, col_name)

        # Calcul de la température de surface (LST)
        sigma = 5.67e-8
        emissivite = 0.98  # Valeur pour Gebesee, ajustable par site si besoin

        df['LST_Calculee'] = ((df['LW_OUT_Consolide'] - (1 - emissivite) * df['LW_IN_Consolide']) / (emissivite * sigma))**0.25 - 273.15

        # Calcul du rayonnement net si SW disponible
        if 'SW_IN_Consolide' in df.columns and 'SW_OUT_Consolide' in df.columns:
            df['Rn_Consolide'] = (
                (df['SW_IN_Consolide'] - df['SW_OUT_Consolide']) + 
                (df['LW_IN_Consolide'] - df['LW_OUT_Consolide'])
            )
            print(f"   ✅ Rayonnement net (Rn) calculé à partir de SW et LW.")
        else:
            print(f"   ⚠️ Rayonnement net (Rn) non calculable (SW manquant).")

        # 5. Colonnes finales
        colonnes_finales = ['LW_IN_Consolide', 'LW_OUT_Consolide', 'LST_Calculee']
        
        # Ajouter les variables météo consolidées disponibles
        for prefix in PREFIXES_METEO.keys():
            col_name = f"{prefix}_Consolide"
            if col_name in df.columns:
                colonnes_finales.append(col_name)
        
        # Ajouter Rn si calculé
        if 'Rn_Consolide' in df.columns:
            colonnes_finales.append('Rn_Consolide')

        df_final = df[colonnes_finales].dropna(how='all')

        if df_final.empty:
            print(f"   ⚠️ Aucune donnée valide trouvée pour {site}.")
            continue

        # Rapport de disponibilité
        print(f"\n   📊 Rapport de disponibilité pour {site} :")
        for col in colonnes_finales:
            n_valid = df_final[col].notna().sum()
            pct = (n_valid / len(df_final)) * 100
            print(f"      {col:25s} : {n_valid:6d} valeurs ({pct:.1f}%)")

        print(f"\n   🏆 Aperçu des valeurs pour {site} :")
        print(df_final.tail(3))

        # 6. Sauvegarde
        chemin_csv = os.path.join(dossier_sortie, f"donnees_icos_{site}.csv")
        df_final.to_csv(chemin_csv)
        print(f"   💾 Fichier sauvegardé : {chemin_csv}")

if __name__ == "__main__":
    main()