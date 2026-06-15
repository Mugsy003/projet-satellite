"""
Transform/Sentinel3/dms_sharpening_sentinel3.py
================================================
Algorithme DMS (Data Mining Sharpening) appliqué aux données Sentinel-3.

Pipeline :
  1. Charger la LST SLSTR à 1 km
  2. Charger les indices optiques Synergy à 300 m
  3. Dégrader les indices de 300 m → 1 km (agrégation spatiale)
  4. Entraîner le modèle ML (RandomForest) : LST_1km ~ f(Indices_1km)
  5. Appliquer le modèle aux indices 300 m → LST Sharpened 300 m
  6. Correction des résidus pour conservation d'énergie

Facteur de sharpening : 1000m / 300m ≈ 3.3x → on utilise un facteur 3.
"""
import os
import glob
import numpy as np
import rioxarray
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.ndimage import zoom
from config import SITES_PILOTES, LOGGER, OUTPUT_DIR


# Facteur d'agrégation : 3 pixels de 300m ≈ 900m ≈ 1 pixel de 1km
FACTEUR_AGREGATION = 3


def aggregate_NxN(matrice_2d, N):
    """Regroupe les pixels par blocs de NxN et calcule la moyenne."""
    h, w = matrice_2d.shape
    h_new = (h // N) * N
    w_new = (w // N) * N
    matrice_coupee = matrice_2d[:h_new, :w_new]
    return matrice_coupee.reshape(h_new // N, N, w_new // N, N).mean(axis=(1, 3))


def load_raster_as_2d(filepath):
    """Charge un fichier TIF et retourne sa matrice 2D + métadonnées."""
    ds = rioxarray.open_rasterio(filepath)
    profile = {
        "crs": ds.rio.crs,
        "transform": ds.rio.transform(),
        "shape": ds.shape
    }
    array_2d = ds.values.squeeze()
    ds.close()
    return array_2d, profile


def calculate_homogeneity_mask(X_dict_hr, N, threshold=0.20):
    """
    Évalue la variance interne des pixels HR au sein de leur pixel parent basse résolution.
    Retourne un masque booléen 1D : True = Homogène, False = Hétérogène.
    """
    premiere_matrice = list(X_dict_hr.values())[0]
    h, w = premiere_matrice.shape
    h_new, w_new = (h // N) * N, (w // N) * N

    cv_total = np.zeros((h_new // N, w_new // N))
    nb_features = len(X_dict_hr)

    for nom, matrice_hr in X_dict_hr.items():
        matrice_coupee = matrice_hr[:h_new, :w_new]
        blocs = matrice_coupee.reshape(h_new // N, N, w_new // N, N)
        mu = blocs.mean(axis=(1, 3))
        sigma = blocs.std(axis=(1, 3))
        cv = sigma / (np.abs(mu) + 1e-8)
        cv_total += cv

    cv_moyen = cv_total / nb_features
    masque_homogene_2d = cv_moyen < threshold
    return masque_homogene_2d.flatten()


def process_dms_s3(nom_site, date_str, dossier_data):
    """Exécute le DMS Sentinel-3 (1km → 300m) pour une date donnée."""
    LOGGER.info(f"\n   📅 DMS S3 pour {date_str} ...")

    prefixe = f"{date_str}_{nom_site}"

    # Fichier thermique (LST à 1km)
    fichier_lst = os.path.join(dossier_data, f"{prefixe}_LST_S3_1km.tif")
    fichier_sortie = os.path.join(dossier_data, f"{prefixe}_LST_S3_Sharpened_DMS_300m.tif")
    fichier_comparaison = os.path.join(dossier_data, f"{prefixe}_Comparaison_DMS_S3.png")

    if not os.path.exists(fichier_lst):
        LOGGER.warning(f"      ❌ Fichier LST introuvable : {fichier_lst}")
        return

    if os.path.exists(fichier_sortie):
        LOGGER.info(f"      ✅ Déjà traité : {fichier_sortie}")
        return

    # Liste des prédicteurs optiques à 300m
    indices_a_charger = ["NDVI", "NDWI", "SAVI", "EVI"]
    
    # ==========================================
    # 1. CHARGEMENT DES DONNÉES
    # ==========================================
    lst_1km, profile_lst = load_raster_as_2d(fichier_lst)
    h_lr, w_lr = lst_1km.shape

    X_dict_hr = {}  # Dictionnaire des prédicteurs à 300m
    for nom_indice in indices_a_charger:
        chemin = os.path.join(dossier_data, f"{prefixe}_S3_{nom_indice}.tif")
        if os.path.exists(chemin):
            array_2d, _ = load_raster_as_2d(chemin)
            X_dict_hr[nom_indice] = array_2d
        else:
            LOGGER.info(f"      ⏭️  {nom_indice} non trouvé, ignoré.")

    if not X_dict_hr:
        LOGGER.error("      ❌ Aucun prédicteur optique trouvé. Annulation.")
        return

    # Récupérer la taille HR de référence
    ref_hr = list(X_dict_hr.values())[0]
    h_hr, w_hr = ref_hr.shape

    # Ajouter les coordonnées spatiales comme prédicteurs
    grille_y, grille_x = np.indices((h_hr, w_hr))
    X_dict_hr['Coord_X'] = grille_x.astype(float)
    X_dict_hr['Coord_Y'] = grille_y.astype(float)

    noms_features = list(X_dict_hr.keys())

    # ==========================================
    # 2. DÉGRADATION À BASSE RÉSOLUTION (~1km)
    # ==========================================
    LOGGER.info(f"      📉 Dégradation des indices de 300m à ~{300*FACTEUR_AGREGATION}m pour l'apprentissage...")

    X_matrice_lr = []
    for f in noms_features:
        pred_lr_2d = aggregate_NxN(X_dict_hr[f], FACTEUR_AGREGATION)
        X_matrice_lr.append(pred_lr_2d.flatten())
    X_matrice_lr = np.column_stack(X_matrice_lr)

    # Il faut que la LST ait les mêmes dimensions que les indices dégradés
    h_lr_agg = (h_hr // FACTEUR_AGREGATION)
    w_lr_agg = (w_hr // FACTEUR_AGREGATION)

    # Recadrer la LST 1km sur la même grille que les indices dégradés
    # La LST 1km a ses propres dimensions, il faut la réinterpoler
    from scipy.ndimage import zoom as scipy_zoom
    if lst_1km.shape != (h_lr_agg, w_lr_agg):
        lst_resized = scipy_zoom(lst_1km,
                                  (h_lr_agg / lst_1km.shape[0], w_lr_agg / lst_1km.shape[1]),
                                  order=1)
    else:
        lst_resized = lst_1km

    y_lr_1d = lst_resized.flatten()

    # ==========================================
    # 3. FILTRAGE PAR HOMOGÉNÉITÉ
    # ==========================================
    masque_valide = np.isfinite(y_lr_1d)
    for i in range(X_matrice_lr.shape[1]):
        masque_valide &= np.isfinite(X_matrice_lr[:, i])

    LOGGER.info("      🧹 Filtrage par homogénéité (seuil < 20% de variance)...")
    X_dict_pour_masque = {k: v for k, v in X_dict_hr.items() if k not in ['Coord_X', 'Coord_Y']}
    masque_homogene = calculate_homogeneity_mask(X_dict_pour_masque, FACTEUR_AGREGATION, threshold=0.20)

    masque_final = masque_valide & masque_homogene
    nb_avant = np.sum(masque_valide)
    nb_apres = np.sum(masque_final)
    LOGGER.info(f"      → Pixels exclus (hétérogènes) : {nb_avant - nb_apres}")

    X_train_data = X_matrice_lr[masque_final]
    y_train_data = y_lr_1d[masque_final]

    if len(y_train_data) < 10:
        LOGGER.error("      ❌ Pas assez de pixels valides pour l'entraînement.")
        return

    # ==========================================
    # 4. ENTRAÎNEMENT DU MODÈLE
    # ==========================================
    X_tr, X_te, y_tr, y_te = train_test_split(X_train_data, y_train_data, test_size=0.2, random_state=42)

    modele = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    modele.fit(X_tr, y_tr)

    # Importance des features
    importances = modele.feature_importances_
    indices_tries = np.argsort(importances)[::-1]
    LOGGER.info("      🏆 Classement des indices :")
    for idx in indices_tries:
        LOGGER.info(f"         - {noms_features[idx]} : {importances[idx]*100:.1f}%")

    y_pred_test = modele.predict(X_te)
    r2 = r2_score(y_te, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_te, y_pred_test))
    LOGGER.info(f"      📊 Précision (à ~1km) : R² = {r2:.3f} | RMSE = {rmse_train:.2f}°C")

    # ==========================================
    # 5. PRÉDICTION À 300m
    # ==========================================
    LOGGER.info("      ✨ Application du modèle sur la grille 300m...")

    X_matrice_hr = np.column_stack([X_dict_hr[f].flatten() for f in noms_features])
    masque_valide_hr = np.all(np.isfinite(X_matrice_hr), axis=1)

    y_pred_hr_1d = np.full(X_matrice_hr.shape[0], np.nan)
    y_pred_hr_1d[masque_valide_hr] = modele.predict(X_matrice_hr[masque_valide_hr])

    lst_sharpened_hr_2d = y_pred_hr_1d.reshape((h_hr, w_hr))

    # ==========================================
    # 6. CORRECTION DES RÉSIDUS
    # ==========================================
    LOGGER.info("      🛠️ Correction des résidus (conservation d'énergie)...")

    lst_sharpened_agg = aggregate_NxN(lst_sharpened_hr_2d, FACTEUR_AGREGATION)
    residus_lr = lst_resized[:h_lr_agg, :w_lr_agg] - lst_sharpened_agg

    residus_hr = zoom(residus_lr, FACTEUR_AGREGATION, order=1)
    h_target = h_lr_agg * FACTEUR_AGREGATION
    w_target = w_lr_agg * FACTEUR_AGREGATION
    residus_hr = residus_hr[:h_target, :w_target]

    lst_corrigee = lst_sharpened_hr_2d[:h_target, :w_target] + residus_hr

    # Vérification
    lst_verif = aggregate_NxN(lst_corrigee, FACTEUR_AGREGATION)
    y_true_v = lst_resized[:h_lr_agg, :w_lr_agg].flatten()
    y_pred_v = lst_verif.flatten()
    m = np.isfinite(y_true_v) & np.isfinite(y_pred_v)
    rmse_energie = np.sqrt(mean_squared_error(y_true_v[m], y_pred_v[m])) if np.sum(m) > 0 else np.nan
    LOGGER.info(f"      ⚖️  RMSE Conservation d'Énergie : {rmse_energie:.5f}°C")

    # ==========================================
    # 7. SAUVEGARDE TIF
    # ==========================================
    # Utiliser un des TIF d'indices 300m comme référence de géoréférencement
    ref_tif = None
    for nom_indice in indices_a_charger:
        chemin = os.path.join(dossier_data, f"{prefixe}_S3_{nom_indice}.tif")
        if os.path.exists(chemin):
            ref_tif = chemin
            break

    if ref_tif:
        ds_ref = rioxarray.open_rasterio(ref_tif)
        ds_out = ds_ref.isel(x=slice(0, w_target), y=slice(0, h_target)).copy()
        ds_out.values = [lst_corrigee]
        ds_out.rio.to_raster(fichier_sortie)
        ds_ref.close()
    else:
        # Fallback : sauvegarder avec le profil de la LST
        from Transform.Sentinel3.processor_sentinel3 import _save_as_tif
        _save_as_tif(lst_corrigee, profile_lst["transform"], str(profile_lst["crs"]), fichier_sortie)

    LOGGER.info(f"      💾 LST Sharpened S3 (300m) sauvegardée : {fichier_sortie}")

    # ==========================================
    # 8. VISUALISATION COMPARATIVE
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    vmin = np.nanpercentile(lst_1km[np.isfinite(lst_1km)], 2) if np.any(np.isfinite(lst_1km)) else 0
    vmax = np.nanpercentile(lst_1km[np.isfinite(lst_1km)], 98) if np.any(np.isfinite(lst_1km)) else 40

    im0 = axes[0].imshow(lst_1km, cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title("LST S3 originale (1km)", fontsize=13)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axis('off')

    im1 = axes[1].imshow(lst_sharpened_hr_2d[:h_target, :w_target], cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"DMS sans résidus (R²={r2:.2f})", fontsize=13)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')

    im2 = axes[2].imshow(lst_corrigee, cmap='magma', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"DMS avec résidus (RMSE énergie={rmse_energie:.3f}°C)", fontsize=13)
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].axis('off')

    fig.suptitle(f"{nom_site} – {date_str} | DMS Sentinel-3 (1km → 300m)", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fichier_comparaison, dpi=200, bbox_inches='tight')
    plt.close()
    LOGGER.info(f"      🖼️  Comparaison visuelle : {fichier_comparaison}")


def main():
    LOGGER.info("========================================")
    LOGGER.info("🚀 DÉMARRAGE DMS SHARPENING (SENTINEL-3 1km → 300m)")
    LOGGER.info("========================================")

    for nom_site in SITES_PILOTES.keys():
        dossier_data = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{nom_site}_S3", "TIF_Data")
        if not os.path.exists(dossier_data):
            continue

        LOGGER.info(f"\n🌍 === Site : {nom_site} ===")

        # Trouver toutes les dates pour lesquelles on a une LST
        fichiers_lst = glob.glob(os.path.join(dossier_data, f"*_{nom_site}_LST_S3_1km.tif"))

        for chemin in fichiers_lst:
            nom_fichier = os.path.basename(chemin)
            # Format : 2023-12-30_Lamasquere_LST_S3_1km.tif
            date_str = nom_fichier.split(f"_{nom_site}")[0]
            process_dms_s3(nom_site, date_str, dossier_data)

    LOGGER.info("\n✅ DMS SHARPENING S3 TERMINÉ.")


if __name__ == "__main__":
    main()
