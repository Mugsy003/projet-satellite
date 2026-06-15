import os
import requests
import pystac_client
import planetary_computer
from config import LOGGER

def connect_to_catalog():
    """Établit et retourne la connexion authentifiée au catalogue STAC."""
    LOGGER.info("🌐 Connexion au catalogue Planetary Computer (Sentinel-3)...")
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace
    )

def search_images_s3_lst(catalog, bbox, time_of_interest, pays):
    """Recherche les images SLSTR LST (Thermique ~1km) de Sentinel-3."""
    search = catalog.search(
        collections=["sentinel-3-slstr-lst-l2-netcdf"],
        bbox=bbox,
        datetime=time_of_interest
    )
    return _execute_search(search, "SLSTR LST", pays)

def search_images_s3_olci(catalog, bbox, time_of_interest, pays):
    """Recherche les images OLCI (Optique ~300m) ou Synergy de Sentinel-3.
    Nous utilisons 'sentinel-3-synergy-syn-l2-netcdf' pour avoir les réflectances de surface.
    """
    search = catalog.search(
        collections=["sentinel-3-synergy-syn-l2-netcdf"],
        bbox=bbox,
        datetime=time_of_interest
    )
    return _execute_search(search, "Synergy Optique", pays)

def _execute_search(search, nom_collection, pays):
    import time
    from pystac_client.exceptions import APIError

    max_retries = 5
    for attempt in range(max_retries):
        try:
            tous_les_items = list(search.items())
            LOGGER.info(f"   🛰️ {len(tous_les_items)} images Sentinel-3 ({nom_collection}) trouvées au total pour {pays}")
            return tous_les_items
        except APIError as e:
            LOGGER.warning(f"   ⚠️ Erreur API STAC: {e}. Tentative {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                LOGGER.error(f"   ❌ Échec de la recherche STAC après {max_retries} tentatives.")
                return []
        except Exception as e:
            LOGGER.error(f"   ❌ Erreur inattendue lors de la recherche: {e}")
            return []
    return []
