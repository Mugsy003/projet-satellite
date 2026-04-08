"""
Extraction/stac_client.py
Module dédié à l'interaction avec le catalogue STAC (Planetary Computer).
Gère la connexion, les requêtes de recherche et les téléchargements HTTP.
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

def search_images(catalog, bbox, time_of_interest, pays, lt=20):
    """
    Recherche les images satellites pour une zone donnée.
    """
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=bbox,
        datetime=time_of_interest,
        query={
            "eo:cloud_cover": {"lt": lt},
            "platform": {"in": ["landsat-8", "landsat-9"]} 
        }
    )
    
    tous_les_items = list(search.items())
    LOGGER.info(f"   Nombre d'images trouvées pour {pays} : {len(tous_les_items)}")
    
    # ⚠️ Logique de sélection
    mes_items = tous_les_items[:3]
        
    for item in mes_items:
        cloud_cover = item.properties.get('eo:cloud_cover', 'Inconnu')
        date_img = item.datetime.strftime("%Y-%m-%d") if item.datetime else "Date inconnue"
        LOGGER.info(f"   - {item.id} | Date: {date_img} | Nuages: {cloud_cover}%")
        
    return mes_items

def download_preview(selected_item, pays, previews_dir):

    if not selected_item or "rendered_preview" not in selected_item.assets:
        LOGGER.warning(f"   ⚠️ Aucune prévisualisation disponible pour {pays}.")
        return
        
    asset_href = selected_item.assets["rendered_preview"].href
    
    # Création du sous-dossier pour le pays
    dossier_pays = os.path.join(previews_dir, pays)
    os.makedirs(dossier_pays, exist_ok=True)
    
    chemin_complet = os.path.join(dossier_pays, f"{selected_item.id}_preview.png")
    
    try:
        response = requests.get(asset_href)
        response.raise_for_status()
        with open(chemin_complet, "wb") as f:
            f.write(response.content)
        LOGGER.info(f"   ✅ Prévisualisation sauvegardée : {chemin_complet}")
    except Exception as e:
        LOGGER.error(f"   ❌ Erreur lors du téléchargement de la prévisualisation: {e}")