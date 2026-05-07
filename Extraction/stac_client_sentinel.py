"""
Extraction/stac_client_sentinel.py
"""
import os
import requests
import pystac_client
import planetary_computer
from config import LOGGER

def connect_to_catalog():
    """Établit et retourne la connexion authentifiée au catalogue STAC."""
    LOGGER.info("🌐 Connexion au catalogue Planetary Computer...")
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

def search_images_s2(catalog, bbox, time_of_interest, pays):
    """
    Recherche TOUTES les images satellites Sentinel-2 pour une zone donnée.
    """
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=time_of_interest
    )
    import time
    from pystac_client.exceptions import APIError

    max_retries = 5
    for attempt in range(max_retries):
        try:
            tous_les_items = list(search.items())
            LOGGER.info(f"   🛰️ {len(tous_les_items)} images Sentinel-2 trouvées au total pour {pays}")
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

def download_preview_s2(selected_item, pays, previews_dir):
    if not selected_item or "rendered_preview" not in selected_item.assets:
        LOGGER.warning(f"   ⚠️ Aucune prévisualisation disponible pour {pays}.")
        return
        
    asset_href = selected_item.assets["rendered_preview"].href
    dossier_pays = os.path.join(previews_dir, pays + "_S2")
    os.makedirs(dossier_pays, exist_ok=True)
    
    chemin_complet = os.path.join(dossier_pays, f"{selected_item.id}_preview.png")
    
    try:
        response = requests.get(asset_href)
        response.raise_for_status()
        with open(chemin_complet, "wb") as f:
            f.write(response.content)
        LOGGER.info(f"   ✅ Prévisualisation Sentinel-2 sauvegardée : {chemin_complet}")
    except Exception as e:
        LOGGER.error(f"   ❌ Erreur téléchargement preview S2: {e}")
