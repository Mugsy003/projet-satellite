import tkinter as tk
from tkinter import simpledialog

def demander_aoi_popup():
    """
    Ouvre une fenêtre pop-up pour demander les coordonnées GPS à l'utilisateur.
    Retourne un tuple (longitude, latitude) ou None si annulé.
    """
    # 1. Créer la fenêtre principale Tkinter
    root = tk.Tk()
    root.withdraw()
    
    # 2. Ouvrir la boîte de dialogue
    reponse = simpledialog.askstring(
        titre="Choix de la Zone d'Intérêt (AOI)",
        invite="Entrez les coordonnées (Longitude, Latitude) séparées par une virgule :\nExemple pour Paris : 2.35, 48.85"
    )
    
    # 3. Traiter la réponse
    if reponse:
        try:
            lon_str, lat_str = reponse.split(",")
            lon = float(lon_str.strip())
            lat = float(lat_str.strip())
            return (lon, lat)
        except ValueError:
            print("⚠️ Erreur : Le format n'est pas respecté (attendu : lon, lat).")
            return None
    else:
        return None