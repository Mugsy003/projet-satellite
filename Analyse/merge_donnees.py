import pandas as pd
import io
import os

def main():
    # Paths
    portugal_path = 'donnees_gol/gol_export_Portugal_caraio_20260304_190301.xlsx'
    italy_path = 'donnees_gol/gol_export_20260310_223705.csv.xlsx'
    greece_path = 'donnees_gol/Gol_Eg_Pyrg_10min_20241128_20250210.txt'
    output_path = 'donnees_pays.csv'

    print("Lecture des données du Portugal...")
    df_portugal = pd.read_excel(portugal_path)
    df_portugal['pays'] = 'Portugal'

    print("Lecture des données de l'Italie...")
    df_italy_raw = pd.read_excel(italy_path)
    # The Italy file is a CSV saved as a single column inside an Excel file
    csv_content = df_italy_raw.columns[0] + '\n' + '\n'.join(df_italy_raw.iloc[:, 0].astype(str))
    df_italy = pd.read_csv(io.StringIO(csv_content), sep=',')
    df_italy['pays'] = 'Italie'

    print("Lecture des données de la Grèce...")
    with open(greece_path, 'r', encoding='utf-8') as f:
        greece_content = f.read()
    
    # Corriger l'absence de retour à la ligne entre le header et la première ligne de données
    # Le fichier lie la dernière colonne au premier timestamp comme : "RHsoil B 60cm %202..."
    greece_content = greece_content.replace('RHsoil B 60cm %202', 'RHsoil B 60cm %\n202')
    
    df_greece = pd.read_csv(io.StringIO(greece_content), sep=';')
    
    # Nettoyage des noms de colonnes pour la Grèce
    df_greece.columns = [str(col).strip() for col in df_greece.columns]

    # Mapping des colonnes pour la Grèce
    col_mapping_greece = {
        'Timestamp': 'created_date',
        'Tair °C': 'field_tair_c_avg',
        'RH %': 'field_rh_avg',
        'WS m/s': 'field_ws_ms',
        'WD °': 'field_wdir',
        'Rain mm': 'field_rain_mm_tot',
        'PAR μmol/m2 s': 'field_par_umol_m2s_avg',
        'LowWaveUp W/m2': 'field_lowwaveup_avg',
        'LowWaveDn W/m2': 'field_lowwavedn_avg',
        'HighWaveUp W/m2': 'field_highwaveup_avg',
        'HighWaveDn W/m2': 'field_highwavedn_avg',
        'LowReflect': 'field_lowreflect_avg',
        'HighReflect': 'field_highreflect_avg',
        'NDVI': 'field_ndvi_avg',
        'Tsoil A 10cm °C': 'field_tsoil_a_10_avg',
        'Tsoil A 25cm °C': 'field_tsoil_a_25_avg',
        'Tsoil B 45cm °C': 'field_tsoil_b_45_avg',
        'Tsoil B 60cm °C': 'field_tsoil_b_60_avg',
        'RHsoil A 10cm %': 'field_rhsoil_a_10_avg',
        'RHsoil A 25cm %': 'field_rhsoil_a_25_avg',
        'RHsoil B 45cm %': 'field_rhsoil_b_45_avg',
        'RHsoil B 60cm %': 'field_rhsoil_b_60_avg'
    }
    
    # Renommer les colonnes
    df_greece.rename(columns=col_mapping_greece, inplace=True)
    df_greece['pays'] = 'Grece'

    print("Concaténation des données...")
    # On rassemble les trois DataFrames
    df_final = pd.concat([df_portugal, df_italy, df_greece], ignore_index=True)
    
    print(f"Dimensions du fichier final: {df_final.shape}")
    
    # Sauvegarde en CSV
    df_final.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Fichier sauvegardé avec succès dans: {output_path}")

if __name__ == '__main__':
    main()
