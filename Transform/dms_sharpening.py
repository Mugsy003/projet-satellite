import os
import glob
import numpy as np
import rioxarray
import xarray as xr
import matplotlib
matplotlib.use('Agg') # Empêche les crashs de fenêtres (Tkinter)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from config import SITES_PILOTES, LOGGER

# --- CONFIGURATION BASE ---
DOSSIER_BASE = r"Outputs"

def aggregate_3x3(matrice_2d):
    """Regroupe les pixels par blocs de 3x3 et calcule la moyenne.
    Permet de repasser d'une grille de 30m à une grille de ~90m."""
    h, w = matrice_2d.shape
    h_new = (h // 3) * 3
    w_new = (w // 3) * 3
    matrice_coupee = matrice_2d[:h_new, :w_new]
    return matrice_coupee.reshape(h_new // 3, 3, w_new // 3, 3).mean(axis=(1, 3))

def load_raster_as_2d(filepath):
    """Charge un fichier TIF et retourne sa matrice 2D (nécessaire pour l'agrégation spatiale)."""
    ds = rioxarray.open_rasterio(filepath)
    profile = {
        "crs": ds.rio.crs,
        "transform": ds.rio.transform(),
        "shape": ds.shape
    }
    array_2d = ds.values.squeeze()
    return array_2d, profile

def process_dms_for_image(nom_site, date_str, dossier_indices):
    """Exécute l'algorithme DMS rigoureux (Apprentissage à 90m, Inférence à 30m)."""
    LOGGER.info(f"\n   📅 Traitement de l'image du {date_str} (Méthode Rigoureuse)...")

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

    # ==========================================
    # 1. CHARGEMENT DES DONNÉES EN 30m
    # ==========================================
    if not os.path.exists(fichier_thermique):
        LOGGER.warning(f"   ❌ Fichier thermique introuvable : {fichier_thermique}")
        return

    lst_30m_2d, raster_profile = load_raster_as_2d(fichier_thermique)
    h, w = raster_profile["shape"][1], raster_profile["shape"][2]
    
    X_dict_30m = {}
    for nom_fichier in liste_predicteurs:
        chemin = os.path.join(dossier_indices, nom_fichier)
        if os.path.exists(chemin):
            array_2d, _ = load_raster_as_2d(chemin)
            X_dict_30m[nom_fichier.split('.')[0]] = array_2d

    if not X_dict_30m:
        LOGGER.error("   ❌ Aucun prédicteur trouvé. Annulation.")
        return
    noms_features = list(X_dict_30m.keys())

    # ==========================================
    # 2. DÉGRADATION À 90m POUR L'APPRENTISSAGE
    # ==========================================
    LOGGER.info("   📉 Dégradation mathématique à 90m pour l'apprentissage physique...")
    
    lst_90m_2d = aggregate_3x3(lst_30m_2d)
    y_90m_1d = lst_90m_2d.flatten()

    X_matrice_90m = []
    for f in noms_features:
        pred_90m_2d = aggregate_3x3(X_dict_30m[f])
        X_matrice_90m.append(pred_90m_2d.flatten())
    X_matrice_90m = np.column_stack(X_matrice_90m)

    # Nettoyage des NaNs sur les données 90m
    masque_valide_90m = np.isfinite(y_90m_1d)
    for i in range(X_matrice_90m.shape[1]):
        masque_valide_90m &= np.isfinite(X_matrice_90m[:, i])

    X_train_90m = X_matrice_90m[masque_valide_90m]
    y_train_90m = y_90m_1d[masque_valide_90m]

    if len(y_train_90m) == 0:
        LOGGER.error("   ❌ Données invalides à 90m (tous les pixels sont NaN).")
        return

    # ==========================================
    # 3. ENTRAÎNEMENT DU MODÈLE (Sur le 90m !)
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_90m, y_train_90m, test_size=0.2, random_state=42
    )

    modele = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    modele.fit(X_train, y_train)

    LOGGER.info("   🏆 Classement des indices :")
    importances = modele.feature_importances_
    indices_tries = np.argsort(importances)[::-1]
    for i in indices_tries:
        LOGGER.info(f"      - {noms_features[i]} : {importances[i] * 100:.1f} %")

    y_test_pred = modele.predict(X_test)
    r2 = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_test, y_test_pred))
    LOGGER.info(f"   📊 Précision Physique (à 90m) : R² = {r2:.3f} | RMSE = {rmse_train:.2f} °C")

    # ==========================================
    # 4. LE SHARPENING (Prédiction sur le 30m)
    # ==========================================
    LOGGER.info("   ✨ Application de la loi thermodynamique sur la grille HD 30m...")
    
    # On aplatit les images 30m d'origine pour faire la prédiction
    y_30m_1d = lst_30m_2d.flatten()
    X_matrice_30m = np.column_stack([X_dict_30m[f].flatten() for f in noms_features])

    masque_valide_30m = np.isfinite(y_30m_1d)
    for i in range(X_matrice_30m.shape[1]):
        masque_valide_30m &= np.isfinite(X_matrice_30m[:, i])

    # L'IA génère la carte HD !
    y_pred_total_30m_1d = np.full_like(y_30m_1d, np.nan) 
    y_pred_total_30m_1d[masque_valide_30m] = modele.predict(X_matrice_30m[masque_valide_30m])
    
    lst_sharpened_30m_2d = y_pred_total_30m_1d.reshape((h, w))

# =====================================================================
    # 🔍 ÉTAPE ULTIME : LA CORRECTION DES RÉSIDUS (BILAN D'ÉNERGIE = 0.0)
    # =====================================================================
    LOGGER.info("   🛠️ Application de la Correction des Résidus (Méthode TsHARP)...")
    
    # On dégrade notre prédiction pour la comparer au vrai satellite
    lst_sharpened_agg_90m = aggregate_3x3(lst_sharpened_30m_2d)

    # 1. Calcul du Résidu (L'erreur du modèle à 90m)
    residus_90m = lst_90m_2d - lst_sharpened_agg_90m
    
    # 2. Redimensionnement du Résidu (On duplique chaque pixel d'erreur en un bloc de 3x3)
    # np.repeat répète les pixels horizontalement et verticalement pour repasser à 30m
    residus_30m = np.repeat(np.repeat(residus_90m, 3, axis=0), 3, axis=1)

    # 3. Correction Finale !
    # On tronque légèrement l'image de base pour qu'elle corresponde parfaitement aux bords de 90m
    h_new, w_new = residus_30m.shape
    lst_sharpened_30m_2d_corrige = lst_sharpened_30m_2d[:h_new, :w_new] + residus_30m

    # --- Vérification que la Magie a opéré ---
    # Si on re-dégrade la NOUVELLE image corrigée à 90m, l'erreur doit être de 0 !
    lst_verif_90m = aggregate_3x3(lst_sharpened_30m_2d_corrige)
    
    y_true_verif = lst_90m_2d.flatten()
    y_pred_verif = lst_verif_90m.flatten()
    masque_verif = np.isfinite(y_true_verif) & np.isfinite(y_pred_verif)

    if np.sum(masque_verif) > 0:
        rmse_energie_final = np.sqrt(mean_squared_error(y_true_verif[masque_verif], y_pred_verif[masque_verif]))
        # Avec cette correction, ce chiffre va être du genre 0.0000000001 °C !
        LOGGER.info(f"   ⚖️  Nouveau RMSE de Conservation d'Énergie : {rmse_energie_final:.5f} °C !")
    # =====================================================================

# =====================================================================
    # SAUVEGARDE TIF (Avec ajustement automatique de la grille GPS)
    # =====================================================================
    ds_base = rioxarray.open_rasterio(fichier_thermique)
    
    # 1. On coupe la carte GPS d'origine pour qu'elle corresponde aux dimensions rognées
    ds_out = ds_base.isel(x=slice(0, w_new), y=slice(0, h_new)).copy()
    
    # 2. On injecte nos nouveaux pixels thermiques HD à l'intérieur
    ds_out.values = [lst_sharpened_30m_2d_corrige]
    
    # 3. Sauvegarde sur le disque
    ds_out.rio.to_raster(fichier_sortie)
    LOGGER.info(f"   💾 TIF HD sauvegardé : {fichier_sortie}")

    # ==========================================
    # 5. SAUVEGARDE VISUELLE (PNG)
    # ==========================================
    plt.figure(figsize=(14, 7))
    
    plt.subplot(1, 2, 1)
    plt.imshow(lst_30m_2d, cmap='magma', vmin=10, vmax=50) 
    plt.title("Avant : Thermique 100m (Interpolé NASA)", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(lst_sharpened_30m_2d, cmap='magma', vmin=10, vmax=50)
    plt.title(f"Après : DMS (R²={r2:.2f}, Énergie RMSE={rmse_energie_final:.2f}°C)", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(fichier_comparaison, dpi=200, bbox_inches='tight')
    plt.close()

def main():
    LOGGER.info("========================================")
    LOGGER.info("🔥 DÉMARRAGE DU DOWNSCALING DMS POUR TOUS LES SITES")
    LOGGER.info("========================================")

    for nom_site in SITES_PILOTES.keys():
        LOGGER.info(f"\n🌍 === Traitement du site : {nom_site} ===")
        
        dossier_indices = os.path.join(DOSSIER_BASE, f"Serie_Temporelle_{nom_site}", "3_Indices", "TIF_Data")
        
        if not os.path.exists(dossier_indices):
            continue
            
        fichiers_thermiques = glob.glob(os.path.join(dossier_indices, f"*_{nom_site}_LST.tif"))
        
        for chemin_fichier in fichiers_thermiques:
            nom_fichier = os.path.basename(chemin_fichier)
            date_str = nom_fichier.split('_')[0] 
            process_dms_for_image(nom_site, date_str, dossier_indices)

    LOGGER.info("\n✅ Traitement DMS terminé pour tous les sites et toutes les dates !")

if __name__ == "__main__":
    main()