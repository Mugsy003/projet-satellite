import numpy as np
from sklearn.ensemble import IsolationForest

def isolation_forest_filter(dn_array, contamination=0.05):
    filtered = np.empty_like(dn_array)
    b, h, l = dn_array.shape
    
    # Aplatissement pour le ML : (Pixels, Bandes)
    array_reshaped = dn_array.transpose(1, 2, 0)
    pixels_flat = array_reshaped.reshape(-1, b)
    
    mask_valid = ~np.isnan(pixels_flat).any(axis=1)
    
    if np.any(mask_valid):
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        predictions = iso.fit_predict(pixels_flat[mask_valid])
        
        final_mask_flat = np.full(pixels_flat.shape[0], 1)
        final_mask_flat[mask_valid] = predictions
        mask_2d = final_mask_flat.reshape(h, l)
        
        for i in range(b):
            band = dn_array[i].copy()
            band[mask_2d == -1] = np.nan # Supprime les outliers
            filtered[i] = band
    else:
        filtered[:] = np.nan
        
    return filtered