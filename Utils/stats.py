import numpy as np

def compter_pourcentage_blancs(image_rgb):
    """
    Calcule le pourcentage de pixels 100% blancs dans une image RGB.
    """
    if image_rgb.dtype.kind == 'f': 
        valeur_blanc = 1.0
    else: 
        valeur_blanc = 255

    masque_blanc = np.all(image_rgb == valeur_blanc, axis=-1)
    nombre_de_blancs = np.sum(masque_blanc) 
    pourcentage_blancs = (nombre_de_blancs / (image_rgb.shape[0] * image_rgb.shape[1])) * 100
    
    return pourcentage_blancs

def serie_temporelle_pourcentage_nuages(img_rgb):
    """
    Calcule la série temporelle du pourcentage de pixels nuageux dans une image RGB.
    """
    pourcentages_nuages = []
    
    for t in range(img_rgb.shape[0]):
        pourcentage = compter_pourcentage_blancs(img_rgb[t])
        pourcentages_nuages.append(pourcentage)
    
    return pourcentages_nuages