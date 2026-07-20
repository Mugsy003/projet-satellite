import cdsapi
import os
import time

# --- CONFIGURATION PROXY FOURNIE PAR L'UTILISATEUR ---
# --- AUCUN PROXY FORCÉ ---
# L'utilisateur a nettoyé ses variables d'environnement.
# -------------------------------------------------------

print("="*60)
print("[ TEST ] TEST DE CONNEXION RAPIDE À COPERNICUS (PROXY TEST)")
print("="*60)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    print("1. Initialisation du client CDS API (avec verify=False)...")
    c = cdsapi.Client(verify=False)
    
    print("2. Envoi d'une toute petite requête (1 heure, 1 variable)...")
    # Requête minimale pour voir si la connexion tient pendant la mise en file d'attente
    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': '2m_temperature',
            'year': '2022',
            'month': '01',
            'day': '01',
            'time': '12:00',
            'area': [50, 10, 49, 11], # Bounding box minuscule
            'format': 'netcdf',
        },
        'test_proxy.nc'
    )
    
    print("\n[ OK ] SUCCÈS ! La connexion a tenu bon et le fichier a été téléchargé.")
    print("Votre configuration proxy / réseau actuel est PARFAITEMENT fonctionnelle pour Copernicus.")
    
    # Nettoyage
    if os.path.exists('test_proxy.nc'):
        os.remove('test_proxy.nc')

except Exception as e:
    print(f"\n[ FAIL ] ÉCHEC. La connexion a été rompue.")
    print(f"Erreur technique : {e}")
    print("\nConclusion : Le proxy ou le réseau coupe toujours les connexions longues.")
