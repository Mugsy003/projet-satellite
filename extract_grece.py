"""
Script dédié à l'extraction propre des données de température
depuis le fichier GOL de la Grèce.

Problèmes identifiés dans le fichier source :
1. Le header est collé à la première ligne de données (pas de retour à la ligne)
2. Les lignes avant le 07/01/2025 ont 22 valeurs (Tsoil A 10cm et RHsoil A 10cm vides)
3. Les lignes après le 07/01/2025 ont 24 valeurs (2 colonnes supplémentaires : CSI_batt, CSI_temp)
"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    input_file = r'donnees_Gol\Gol_Eg_Pyrg_10min_20241128_20250210.txt'
    output_file = 'temperatures_Grece.csv'

    print(f"Lecture du fichier brut : {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw = f.read()

    # --- Étape 1 : Séparer le header de la première ligne de données ---
    raw = raw.replace('RHsoil B 60cm %2', 'RHsoil B 60cm %\n2', 1)

    lines = raw.split('\n')
    header_line = lines[0].strip()
    base_headers = header_line.split(';')
    print(f"Colonnes du header : {len(base_headers)}")

    # Header étendu pour les lignes à 24 valeurs
    extended_headers = base_headers + ['CSI_batt_min', 'CSI_temp']

    # --- Étape 2 : Parser les lignes de données ---
    data_rows = []
    stats = {'22_vals': 0, '24_vals': 0, 'skipped': 0}

    for i, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue
        values = line.split(';')
        n = len(values)

        if n == 22:
            # Lignes courtes : ajouter les 2 colonnes manquantes comme NaN
            values += ['', '']
            stats['22_vals'] += 1
        elif n == 24:
            stats['24_vals'] += 1
        else:
            stats['skipped'] += 1
            continue

        data_rows.append(values)

    print(f"Lignes à 22 valeurs : {stats['22_vals']}")
    print(f"Lignes à 24 valeurs : {stats['24_vals']}")
    print(f"Lignes ignorées     : {stats['skipped']}")

    df = pd.DataFrame(data_rows, columns=extended_headers)

    # --- Étape 3 : Convertir les types ---
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Colonnes de température à extraire
    temp_cols_src = [
        'Timestamp',
        'Tair °C',
        'Tsoil A 10cm °C',
        'Tsoil A 25cm °C',
        'Tsoil B 45cm °C',
        'Tsoil B 60cm °C',
    ]

    df_temp = df[temp_cols_src].copy()

    # Convertir en numérique
    for col in temp_cols_src:
        if col != 'Timestamp':
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

    # Supprimer les lignes sans timestamp
    df_temp = df_temp.dropna(subset=['Timestamp'])

    # Renommer pour cohérence avec les autres fichiers de températures
    df_temp.rename(columns={
        'Timestamp': 'created_date',
        'Tair °C': 'field_tair_c_avg',
        'Tsoil A 10cm °C': 'field_tsoil_a_10_avg',
        'Tsoil A 25cm °C': 'field_tsoil_a_25_avg',
        'Tsoil B 45cm °C': 'field_tsoil_b_45_avg',
        'Tsoil B 60cm °C': 'field_tsoil_b_60_avg',
    }, inplace=True)

    # Trier par date
    df_temp = df_temp.sort_values('created_date').reset_index(drop=True)

    print(f"\nAperçu des premières lignes :")
    print(df_temp.head(10).to_string())
    print(f"\nAperçu des dernières lignes :")
    print(df_temp.tail(5).to_string())
    print(f"\nPlage temporelle : {df_temp['created_date'].min()} → {df_temp['created_date'].max()}")
    print(f"Dimensions finales : {df_temp.shape}")
    print(f"Valeurs non-NaN par colonne :")
    print(df_temp.notna().sum().to_string())

    # --- Étape 4 : Sauvegarde ---
    df_temp.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✅ Fichier sauvegardé : {output_file}")


if __name__ == '__main__':
    main()
