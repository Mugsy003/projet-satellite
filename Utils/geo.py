import math

def get_bbox_from_point(lon, lat, radius_km=3):
    """
    Calcule une Bounding Box de 'radius_km' autour d'un point GPS.
    Retourne [min_lon, min_lat, max_lon, max_lat]
    """
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / (111.32 * math.cos(math.radians(lat)))
    
    return [lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta]