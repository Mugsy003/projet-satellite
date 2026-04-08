import math

def get_bbox_from_point(lon, lat, radius_km=30):
    """
    Calcule une Bounding Box de 'radius_km' autour d'un point GPS.
    Retourne [min_lon, min_lat, max_lon, max_lat]
    """
    # 1 degré de latitude ~ 111.32 km
    lat_delta = radius_km / 111.32
    # 1 degré de longitude dépend de la latitude (cosinus)
    lon_delta = radius_km / (111.32 * math.cos(math.radians(lat)))
    
    return [lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta]