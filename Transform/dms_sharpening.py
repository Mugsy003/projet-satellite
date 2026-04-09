import os
import glob
import numpy as np
import rioxarray
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from config import SITES_PILOTES, LOGGER

# --- CONFIGURATION BASE ---
DOSSIER_BASE = r"Outputs"

def load_raster_as_array(filepath):
    """Charge un fichier TIF et l'aplatit en vecteur 1D."""
    ds = rioxarray.open_rasterio(filepath)
    profile = {
        "crs": ds.rio.crs,
        "transform": ds.rio.transform(),
        "shape": ds.shape
    }
    array_1d = ds.values.squeeze().flatten()
    return array_1d, profile

def process_dms_for_image(nom_site, date_str, dossier_indices):
    """Exécute l'algorithme DMS pour une date et un site précis."""
    LOGGER.info(f"\n   📅 Traitement de l'image du {date_str}...")

    # 1. Construction dynamique des noms de fichiers
    fichier_thermique = os.path.join(dossier_indices, f"{date_str}_{nom_site}_LST.tif")
    fichier_sortie = os.path.join(dossier_indices, f"{date_str}_{nom_site}_LST_Sharpened_DMS.tif")
    fichier_comparaison = os.path.join(dossier_indices, f"{date_str}_{nom_site}_Comparaison_DMS.png")

    liste_predicteurs = [
        f"{date_str}_{nom_site}_NDVI.tif", 
        f"{date_str}_{nom_site}_NDWI.tif", 
        f"{date_str}_{nom_site}_NDBI.tif", 
        f"{date_str}_{nom_site}_EVI.tif",
        f"{nom_site}_MNT.tif"
    ]

    # 2. CHARGEMENT
    if not os.path.exists(fichier_thermique):
        LOGGER.warning(f"   ❌ Fichier thermique introuvable : {fichier_thermique}")
        return

    y_thermique_1d, raster_profile = load_raster_as_array(fichier_thermique)
    
    X_dict = {}
    for nom_fichier in liste_predicteurs:
        chemin = os.path.join(dossier_indices, nom_fichier)
        if os.path.exists(chemin):
            array_1d, _ = load_raster_as_array(chemin)
            X_dict[nom_fichier.split('.')[0]] = array_1d
        else:
            LOGGER.warning(f"   ⚠️ Prédicteur manquant ignoré : {nom_fichier}")

    if not X_dict:
        LOGGER.error("   ❌ Aucun prédicteur trouvé. Annulation.")
        return

    # 3. MATRICE D'APPRENTISSAGE ET NETTOYAGE
    noms_features = list(X_dict.keys())
    X_matrice = np.column_stack([X_dict[f] for f in noms_features])

    masque_valide = np.isfinite(y_thermique_1d)
    for i in range(X_matrice.shape[1]):
        masque_valide &= np.isfinite(X_matrice[:, i])

    X_propre = X_matrice[masque_valide]
    y_propre = y_thermique_1d[masque_valide]

    if len(y_propre) == 0:
        LOGGER.error("   ❌ Données invalides (tous les pixels sont NaN).")
        return

    # 4. ENTRAÎNEMENT DU MODÈLE (Train/Test Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_propre, y_propre, test_size=0.2, random_state=42
    )

    modele = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    modele.fit(X_train, y_train)

    # Affichage de l'utilité des indices
    importances = modele.feature_importances_
    indices_tries = np.argsort(importances)[::-1]
    
    LOGGER.info("   🏆 Classement des indices :")
    for i in indices_tries:
        LOGGER.info(f"      - {noms_features[i]} : {importances[i] * 100:.1f} %")

    # Évaluation
    y_test_pred = modele.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    LOGGER.info(f"   📊 Précision : R² = {r2:.3f} | RMSE = {rmse:.2f} °C")

    # 5. PRÉDICTION ET SAUVEGARDE TIF
    y_pred_total_1d = np.full_like(y_thermique_1d, np.nan) 
    y_pred_total_1d[masque_valide] = modele.predict(X_propre)

    h, w = raster_profile["shape"][1], raster_profile["shape"][2]
    image_sharpened_2d = y_pred_total_1d.reshape((h, w))

    ds_base = rioxarray.open_rasterio(fichier_thermique)
    ds_out = xr.DataArray(
        [image_sharpened_2d], 
        coords=ds_base.coords,
        dims=ds_base.dims,
        attrs=ds_base.attrs
    )
    ds_out.rio.to_raster(fichier_sortie)
    LOGGER.info(f"   💾 TIF HD sauvegardé : {fichier_sortie}")

    # 6. SAUVEGARDE VISUELLE (PNG)
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(y_thermique_1d.reshape((h, w)), cmap='magma', vmin=20, vmax=50) 
    plt.title("Avant : Thermique 100m", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_sharpened_2d, cmap='magma', vmin=20, vmax=50)
    plt.title(f"Après : DMS (R²={r2:.2f}, RMSE={rmse:.2f}°C)", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(fichier_comparaison, dpi=200, bbox_inches='tight')
    plt.close() # Indispensable pour ne pas saturer la RAM dans une boucle !

def main():
    LOGGER.info("========================================")
    LOGGER.info("🔥 DÉMARRAGE DU DOWNSCALING DMS POUR TOUS LES SITES")
    LOGGER.info("========================================")

    # On boucle sur chaque site défini dans le config.py
    for nom_site in SITES_PILOTES.keys():
        LOGGER.info(f"\n🌍 === Traitement du site : {nom_site} ===")
        
        dossier_indices = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}", "3_Indices", "TIF_Data")
        
        if not os.path.exists(dossier_indices):
            LOGGER.warning(f"⚠️ Dossier introuvable pour {nom_site}. Ignoré.")
            continue
            
        # On liste dynamiquement toutes les images thermiques présentes dans le dossier
        fichiers_thermiques = glob.glob(os.path.join(dossier_indices, f"*_{nom_site}_LST.tif"))
        
        if not fichiers_thermiques:
            LOGGER.warning(f"⚠️ Aucune image thermique à traiter pour {nom_site}.")
            continue
            
        # Pour chaque image thermique trouvée, on extrait la date et on lance l'algo
        for chemin_fichier in fichiers_thermiques:
            nom_fichier = os.path.basename(chemin_fichier)
            # Extrait la date "YYYY-MM-DD" au début du nom de fichier
            date_str = nom_fichier.split('_')[0] 
            
            process_dms_for_image(nom_site, date_str, dossier_indices)

    LOGGER.info("\n✅ Traitement DMS terminé pour tous les sites et toutes les dates !")

if __name__ == "__main__":
    main()