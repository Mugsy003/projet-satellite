"""
Extraction/stac_client_ecostress.py
"""
import os
import requests
import pystac_client
from config import LOGGER

def connect_to_catalog():
    """Établit et retourne la connexion au catalogue STAC NASA CMR (LPCLOUD)."""
    LOGGER.info("🌐 Connexion au catalogue NASA CMR (LPCLOUD) pour ECOSTRESS...")
    return pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")

def search_images_ecostress(catalog, bbox, time_of_interest, pays):
    """
    Recherche TOUTES les images satellites ECOSTRESS pour une zone donnée.
    Collection: ECO_L2T_LSTE.v002
    """
    search = catalog.search(
        collections=["ECO_L2T_LSTE.v002"],
        bbox=bbox,
        datetime=time_of_interest
    )
    import time
    from pystac_client.exceptions import APIError

    max_retries = 5
    for attempt in range(max_retries):
        try:
            tous_les_items = list(search.items())
            LOGGER.info(f"   🛰️ {len(tous_les_items)} images ECOSTRESS trouvées au total pour {pays}")
            return tous_les_items
        except APIError as e:
            LOGGER.warning(f"   ⚠️ Erreur API STAC: {e}. Tentative {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                LOGGER.error(f"   ❌ Échec de la recherche STAC après {max_retries} tentatives.")
                return []
        except Exception as e:
            LOGGER.error(f"   ❌ Erreur inattendue lors de la recherche: {e}")
            return []

    return []

def download_preview_ecostress(selected_item, pays, previews_dir):
    # Les previews pour ECOSTRESS sont souvent sous la clé "browse"
    browse_key = None
    for key in selected_item.assets.keys():
        if "browse" in key.lower():
            browse_key = key
            break
            
    if not browse_key:
        LOGGER.warning(f"   ⚠️ Aucune prévisualisation disponible pour {pays} (ECOSTRESS).")
        return
        
    asset_href = selected_item.assets[browse_key].href
    dossier_pays = os.path.join(previews_dir, pays + "_ECOSTRESS")
    os.makedirs(dossier_pays, exist_ok=True)
    
    chemin_complet = os.path.join(dossier_pays, f"{selected_item.id}_preview.png")
    
    try:
        response = requests.get(asset_href)
        response.raise_for_status()
        with open(chemin_complet, "wb") as f:
            f.write(response.content)
        LOGGER.info(f"   ✅ Prévisualisation ECOSTRESS sauvegardée : {chemin_complet}")
    except Exception as e:
        LOGGER.error(f"   ❌ Erreur téléchargement preview ECOSTRESS: {e}")
