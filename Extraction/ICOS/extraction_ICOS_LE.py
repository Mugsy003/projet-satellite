import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
import numpy as np

try:
    from icoscp.dobj import Dobj
except ImportError:
    print("Erreur: Impossible d'importer 'icoscp'.")
    sys.exit(1)

from config import PIDS_ICOS_L2_FLUXES

def _find_columns_for_prefix(all_columns, prefix):
    """
    Cherche toutes les colonnes ICOS correspondant à un préfixe donné.
    Ex: prefix='LE' trouvera 'LE_F', 'LE_F_MDS', 'LE', etc.
    """
    candidates = sorted([c for c in all_columns if c == prefix or c.startswith(prefix + '_')])
    return candidates

def _consolidate_columns(df, cols, name):
    """
    Fusionne plusieurs colonnes capteurs en une seule colonne consolidée.
    Priorité : colonnes gap-filled (_F) > colonnes brutes.
    """
    if not cols:
        return pd.Series(np.nan, index=df.index, name=name)
    
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
    print("Initialisation de l'extraction des flux (Chaleur Latente - LE) ICOS...")

    dossier_sortie = "Outputs_ICOS"
    os.makedirs(dossier_sortie, exist_ok=True)

    for site, pid in PIDS_ICOS_L2_FLUXES.items():
        print(f"\n========================================")
        print(f"Traitement du site : {site}")
        
        try:
            dobj = Dobj(pid)
        except Exception as e:
            print(f"Erreur lors de l'acces au PID {pid}: {e}")
            continue

        if not dobj.valid:
            print(f"Objet invalide pour {site}.")
            continue

        print(f"Objet charge : {dobj.id}")

        toutes_les_colonnes = dobj.colNames
        if not toutes_les_colonnes:
            print(f"Aucune colonne trouvee pour {site}.")
            continue

        # Recherche de LE
        le_cols = _find_columns_for_prefix(toutes_les_colonnes, "LE")
        print(f"   Capteurs LE trouves : {le_cols}")
        
        # Optionnel: On peut aussi récupérer H (Chaleur Sensible)
        h_cols = _find_columns_for_prefix(toutes_les_colonnes, "H")
        print(f"   Capteurs H trouves : {h_cols}")

        if not le_cols:
            print(f"   LE NON DISPONIBLE pour ce PID.")
            continue

        # Extraction des données
        try:
            df = dobj.data
        except Exception as e:
            print(f"Erreur lors de l'extraction des donnees pour {site}: {e}")
            continue

        # Traitement temporel
        if 'TIMESTAMP' in df.columns:
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
            df.set_index('TIMESTAMP', inplace=True)
        elif 'TIMESTAMP_START' in df.columns:
            # Parfois les datasets Level 2 utilisent TIMESTAMP_START / END
            df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP_START'])
            df.set_index('TIMESTAMP', inplace=True)
        else:
            print("   Impossible de trouver une colonne temporelle.")
            continue

        # Nettoyage des valeurs invalides ICOS
        df.replace([-9.99, -999.0, -9999.0], np.nan, inplace=True)

        # Consolidation des colonnes
        df['LE_Consolide'] = _consolidate_columns(df, le_cols, 'LE_Consolide')
        if h_cols:
            df['H_Consolide'] = _consolidate_columns(df, h_cols, 'H_Consolide')

        colonnes_finales = ['LE_Consolide']
        if 'H_Consolide' in df.columns:
            colonnes_finales.append('H_Consolide')

        df_final = df[colonnes_finales].dropna(how='all')

        if df_final.empty:
            print(f"   Aucune donnee LE valide trouvee pour {site}.")
            continue

        # Rapport de disponibilité
        print(f"\n   Rapport de disponibilite pour {site} :")
        for col in colonnes_finales:
            n_valid = df_final[col].notna().sum()
            pct = (n_valid / len(df_final)) * 100
            print(f"      {col:20s} : {n_valid:6d} valeurs ({pct:.1f}%)")

        print(f"\n   Apercu des valeurs pour {site} :")
        print(df_final.tail(3))

        # Sauvegarde
        nom_fichier = f"donnees_icos_LE_{site}.csv"
        chemin_csv = os.path.join(dossier_sortie, nom_fichier)
        df_final.to_csv(chemin_csv)
        print(f"   Fichier sauvegarde : {chemin_csv}")

if __name__ == "__main__":
    main()
