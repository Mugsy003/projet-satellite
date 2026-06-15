"""
Transform/Fusion/dms_sharpening_s3_s2.py
=========================================
Algorithme DMS Fusion : Combine la thermique Sentinel-3 (1 km)
avec l'optique ultra-haute résolution Sentinel-2 (10 m).

Pipeline :
  1. Charger la LST Sentinel-3 à 1 km (fichier TIF reprojeté)
  2. Charger les indices optiques Sentinel-2 à 10 m (NDVI, NDWI, SAVI, EVI)
  3. Dégrader les indices S2 de 10 m → 1 km (agrégation par blocs de 100x100)
  4. Entraîner le modèle ML : LST_1km ~ f(Indices_1km)
  5. Appliquer le modèle aux indices 10 m → LST Sharpened 10 m
  6. Correction des résidus pour conservation d'énergie

Facteur de sharpening : 1000m / 10m = 100x
Note : Un facteur 100 est très élevé, on agrège plutôt par pas intermédiaires.
"""
import os
import glob
import json
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


# 1km ≈ 100 pixels de 10m, mais on utilise un facteur adaptable
# On agrège par 50 pour avoir ~500m de résolution d'entraînement (compromis)
FACTEUR_AGREGATION = 50


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
    """Masque d'homogénéité pour filtrer les pixels mixtes."""
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
    return (cv_moyen < threshold).flatten()


def process_fusion_s3_s2(nom_site, date_str, dossier_s3, dossier_s2):
    """Exécute le DMS Fusion S3+S2 pour une date donnée."""
    LOGGER.info(f"\n   📅 Fusion S3+S2 pour {date_str} ...")

    prefixe_s3 = f"{date_str}_{nom_site}"

    # Fichiers
    fichier_lst_s3 = os.path.join(dossier_s3, f"{prefixe_s3}_LST_S3_1km.tif")
    fichier_sortie = os.path.join(dossier_s3, f"{prefixe_s3}_LST_Fusion_S3_S2_10m.tif")
    fichier_comparaison = os.path.join(dossier_s3, f"{prefixe_s3}_Comparaison_Fusion_S3_S2.png")

    if not os.path.exists(fichier_lst_s3):
        LOGGER.warning(f"      ❌ LST S3 introuvable : {fichier_lst_s3}")
        return

    if os.path.exists(fichier_sortie):
        LOGGER.info(f"      ✅ Déjà traité : {fichier_sortie}")
        return

    # Chercher les indices S2 pour la même date (ou date très proche)
    # Les fichiers S2 sont du type : 2023-12-30_10h42_Lamasquere_NDVI.tif
    indices_s2 = {}
    for nom_indice in ["NDVI", "NDWI", "SAVI", "EVI"]:
        pattern = os.path.join(dossier_s2, f"{date_str[:10]}*_{nom_site}_{nom_indice}.tif")
        matches = glob.glob(pattern)
        if matches:
            indices_s2[nom_indice] = matches[0]

    if not indices_s2:
        LOGGER.warning(f"      ⚠️ Aucun indice S2 trouvé pour {date_str}. Skip.")
        return

    LOGGER.info(f"      📋 Indices S2 trouvés : {list(indices_s2.keys())}")

    # ==========================================
    # 1. CHARGEMENT
    # ==========================================
    lst_1km, profile_lst = load_raster_as_2d(fichier_lst_s3)

    X_dict_hr = {}
    for nom_indice, chemin in indices_s2.items():
        array_2d, _ = load_raster_as_2d(chemin)
        X_dict_hr[nom_indice] = array_2d

    ref_hr = list(X_dict_hr.values())[0]
    h_hr, w_hr = ref_hr.shape

    # Coordonnées spatiales
    grille_y, grille_x = np.indices((h_hr, w_hr))
    X_dict_hr['Coord_X'] = grille_x.astype(float)
    X_dict_hr['Coord_Y'] = grille_y.astype(float)
    noms_features = list(X_dict_hr.keys())

    # ==========================================
    # 2. DÉGRADATION À ~500m
    # ==========================================
    LOGGER.info(f"      📉 Dégradation S2 de 10m à ~{10*FACTEUR_AGREGATION}m...")

    X_matrice_lr = []
    for f in noms_features:
        pred_lr = aggregate_NxN(X_dict_hr[f], FACTEUR_AGREGATION)
        X_matrice_lr.append(pred_lr.flatten())
    X_matrice_lr = np.column_stack(X_matrice_lr)

    h_lr_agg = h_hr // FACTEUR_AGREGATION
    w_lr_agg = w_hr // FACTEUR_AGREGATION

    # Redimensionner la LST 1km pour correspondre à la grille dégradée
    from scipy.ndimage import zoom as scipy_zoom
    lst_resized = scipy_zoom(lst_1km,
                              (h_lr_agg / lst_1km.shape[0], w_lr_agg / lst_1km.shape[1]),
                              order=1)
    y_lr_1d = lst_resized.flatten()

    # ==========================================
    # 3. FILTRAGE + ENTRAÎNEMENT
    # ==========================================
    masque_valide = np.isfinite(y_lr_1d)
    for i in range(X_matrice_lr.shape[1]):
        masque_valide &= np.isfinite(X_matrice_lr[:, i])

    X_dict_pour_masque = {k: v for k, v in X_dict_hr.items() if k not in ['Coord_X', 'Coord_Y']}
    masque_homogene = calculate_homogeneity_mask(X_dict_pour_masque, FACTEUR_AGREGATION, threshold=0.20)
    masque_final = masque_valide & masque_homogene

    LOGGER.info(f"      → Pixels d'entraînement valides : {np.sum(masque_final)}")

    X_train_data = X_matrice_lr[masque_final]
    y_train_data = y_lr_1d[masque_final]

    if len(y_train_data) < 10:
        LOGGER.error("      ❌ Pas assez de pixels valides. Annulation.")
        return

    X_tr, X_te, y_tr, y_te = train_test_split(X_train_data, y_train_data, test_size=0.2, random_state=42)

    modele = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    modele.fit(X_tr, y_tr)

    y_pred_test = modele.predict(X_te)
    r2 = r2_score(y_te, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred_test))
    LOGGER.info(f"      📊 R² = {r2:.3f} | RMSE = {rmse:.2f}°C")

    # ==========================================
    # 4. PRÉDICTION À 10m
    # ==========================================
    LOGGER.info("      ✨ Application sur la grille S2 (10m)...")

    X_matrice_hr = np.column_stack([X_dict_hr[f].flatten() for f in noms_features])
    masque_valide_hr = np.all(np.isfinite(X_matrice_hr), axis=1)

    y_pred_hr = np.full(X_matrice_hr.shape[0], np.nan)
    y_pred_hr[masque_valide_hr] = modele.predict(X_matrice_hr[masque_valide_hr])
    lst_sharpened = y_pred_hr.reshape((h_hr, w_hr))

    # ==========================================
    # 5. CORRECTION DES RÉSIDUS
    # ==========================================
    LOGGER.info("      🛠️ Correction des résidus...")

    lst_agg = aggregate_NxN(lst_sharpened, FACTEUR_AGREGATION)
    residus = lst_resized[:h_lr_agg, :w_lr_agg] - lst_agg
    residus_hr = zoom(residus, FACTEUR_AGREGATION, order=1)

    h_target = h_lr_agg * FACTEUR_AGREGATION
    w_target = w_lr_agg * FACTEUR_AGREGATION
    residus_hr = residus_hr[:h_target, :w_target]
    lst_corrigee = lst_sharpened[:h_target, :w_target] + residus_hr

    rmse_energie = np.nan
    lst_verif = aggregate_NxN(lst_corrigee, FACTEUR_AGREGATION)
    y_t = lst_resized[:h_lr_agg, :w_lr_agg].flatten()
    y_p = lst_verif.flatten()
    m = np.isfinite(y_t) & np.isfinite(y_p)
    if np.sum(m) > 0:
        rmse_energie = np.sqrt(mean_squared_error(y_t[m], y_p[m]))
    LOGGER.info(f"      ⚖️  RMSE Conservation d'Énergie : {rmse_energie:.5f}°C")

    # ==========================================
    # 6. SAUVEGARDE
    # ==========================================
    ref_tif = list(indices_s2.values())[0]
    ds_ref = rioxarray.open_rasterio(ref_tif)
    ds_out = ds_ref.isel(x=slice(0, w_target), y=slice(0, h_target)).copy()
    ds_out.values = [lst_corrigee]
    ds_out.rio.to_raster(fichier_sortie)
    ds_ref.close()
    LOGGER.info(f"      💾 LST Fusion S3+S2 (10m) : {fichier_sortie}")

    # ==========================================
    # 7. VISUALISATION
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    vmin = np.nanpercentile(lst_1km[np.isfinite(lst_1km)], 2) if np.any(np.isfinite(lst_1km)) else 0
    vmax = np.nanpercentile(lst_1km[np.isfinite(lst_1km)], 98) if np.any(np.isfinite(lst_1km)) else 40

    im0 = axes[0].imshow(lst_1km, cmap='magma', vmin=vmin, vmax=vmax)
    axes[0].set_title("LST S3 originale (1km)", fontsize=13)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].axis('off')

    im1 = axes[1].imshow(lst_corrigee, cmap='magma', vmin=vmin, vmax=vmax)
    axes[1].set_title(f"DMS Fusion S3+S2 10m (R²={r2:.2f})", fontsize=13)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')

    fig.suptitle(f"{nom_site} – {date_str} | Fusion S3+S2 (1km → 10m)", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fichier_comparaison, dpi=200, bbox_inches='tight')
    plt.close()


def main():
    LOGGER.info("========================================")
    LOGGER.info("🚀 DÉMARRAGE DMS FUSION (SENTINEL-3 1km + SENTINEL-2 10m → 10m)")
    LOGGER.info("========================================")

    for nom_site in SITES_PILOTES.keys():
        # Chercher le manifeste de paires S3/S2
        chemin_manifeste = os.path.join(OUTPUT_DIR, f"manifest_paires_S3_S2_{nom_site}.json")
        if not os.path.exists(chemin_manifeste):
            continue

        dossier_s3 = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{nom_site}_S3", "TIF_Data")
        dossier_s2 = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{nom_site}_S2", "3_Indices", "TIF_Data")

        if not os.path.exists(dossier_s3) or not os.path.exists(dossier_s2):
            LOGGER.info(f"   ⏭️  Dossiers S3 ou S2 manquants pour {nom_site}. Skip.")
            continue

        LOGGER.info(f"\n🌍 === Site : {nom_site} ===")

        with open(chemin_manifeste, "r") as f:
            paires = json.load(f)

        for paire in paires:
            date_str = paire["date"]
            process_fusion_s3_s2(nom_site, date_str, dossier_s3, dossier_s2)

    LOGGER.info("\n✅ DMS FUSION S3+S2 TERMINÉ.")


if __name__ == "__main__":
    main()
