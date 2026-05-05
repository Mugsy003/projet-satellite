import pandas as pd
import os

def extract_temperatures():
    input_file = 'donnees_pays.csv'
    print(f"Lecture du fichier {input_file}...")
    df = pd.read_csv(input_file)
    
    # Sélectionner les colonnes de température (Air et Sol) + la date
    cols_to_keep = [
        'created_date',
        'field_tair_c_avg',
        'field_tsoil_a_10_avg',
        'field_tsoil_a_20_avg',
        'field_tsoil_a_25_avg',
        'field_tsoil_a_40_avg',
        'field_tsoil_b_45_avg',
        'field_tsoil_b_60_avg'
    ]
    
    # Garder uniquement les colonnes existantes dans le DataFrame
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    
    # On garde 'pays' pour filtrer
    df_temp = df[existing_cols + ['pays']].copy()
    
    # Pour chaque pays, on sauvegarde dans un fichier csv séparé
    for pays in df_temp['pays'].dropna().unique():
        # Filtrer par pays et supprimer la colonne 'pays' du rendu final
        df_pays = df_temp[df_temp['pays'] == pays].drop(columns=['pays'])
        
        output_filename = f'temperatures_{pays}.csv'
        df_pays.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"Fichier créé avec succès : {output_filename} ({len(df_pays)} lignes, {len(df_pays.columns)} colonnes)")

if __name__ == '__main__':
    extract_temperatures()
