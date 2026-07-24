"""
Module PT-SINRH : Modèle d'Évapotranspiration Priestley-Taylor
avec contraintes éco-physiologiques (Fisher et al., 2008).

Ce module implémente une version vectorisée du modèle PT-JPL adaptée
pour traiter des cubes de données spatio-temporelles (temps, y, x).

Référence :
    Fisher, J.B., Tu, K.P. & Baldocchi, D.D. (2008).
    Global estimates of the land–atmosphere water flux based on monthly
    AVHRR and ISLSCP-II data, validated at 16 FLUXNET sites.
    Remote Sensing of Environment, 112(3), 901-919.
"""

import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

# ===========================================================================
# CONSTANTES DU MODÈLE
# ===========================================================================

ALPHA_PT   = 1.26    # Coefficient de Priestley-Taylor (sans unité)
K_RN       = 0.6     # Coefficient d'extinction du rayonnement net à travers la canopée
K_PAR      = 0.5     # Coefficient d'extinction du PAR (Beer-Lambert)
M1         = 1.3632  # Pente de la relation linéaire SAVI → fAPAR
B1         = -0.048  # Ordonnée à l'origine de la relation SAVI → fAPAR
M2         = 1.0     # Pente de la relation linéaire NDVI → fIPAR
B2         = -0.05   # Ordonnée à l'origine de la relation NDVI → fIPAR
GAMMA_PSY  = 0.066   # Constante psychrométrique (kPa·°C⁻¹)


def calculate_pt_sinrh_et(
    RH: np.ndarray,
    Rn: np.ndarray,
    G: np.ndarray,
    T_max: np.ndarray,
    NDVI: np.ndarray,
    SAVI: np.ndarray,
    Delta: np.ndarray,
    VPD: np.ndarray,
    PAR: np.ndarray,
) -> dict:
    """
    Calcule l'évapotranspiration (ET) selon le modèle PT-SINRH.

    Le modèle décompose l'ET en trois composantes physiques :
      - ET_c  : Transpiration de la canopée (contrôlée par les stomates).
      - ET_s  : Évaporation du sol (contrôlée par l'humidité de surface).
      - ET_i  : Évaporation de l'eau interceptée par les feuilles.

    Chaque composante est modulée par des contraintes éco-physiologiques
    (f_wet, f_g, f_M, f_T, f_SM) dérivées de la télédétection et de la
    météorologie.

    Paramètres
    ----------
    RH : np.ndarray, shape (T, Y, X)
        Humidité relative, normalisée entre 0 et 1.
    Rn : np.ndarray, shape (T, Y, X)
        Rayonnement net à la surface (W·m⁻²).
    G : np.ndarray, shape (T, Y, X)
        Flux de chaleur dans le sol (W·m⁻²).
    T_max : np.ndarray, shape (T, Y, X)
        Température maximale journalière de l'air (°C).
    NDVI : np.ndarray, shape (T, Y, X)
        Indice de végétation par différence normalisée (sans unité).
    SAVI : np.ndarray, shape (T, Y, X)
        Indice de végétation ajusté pour le sol (sans unité).
    Delta : np.ndarray, shape (T, Y, X)
        Pente de la courbe de pression de vapeur saturante (kPa·°C⁻¹).
    VPD : np.ndarray, shape (T, Y, X)
        Déficit de pression de vapeur (kPa).
    PAR : np.ndarray, shape (T, Y, X)
        Rayonnement photosynthétiquement actif (W·m⁻² ou µmol·m⁻²·s⁻¹).

    Retourne
    --------
    dict
        Dictionnaire contenant les clés suivantes :
        - 'ET'    : np.ndarray (T, Y, X) – ET totale (W·m⁻²).
        - 'ET_c'  : np.ndarray (T, Y, X) – Transpiration de la canopée.
        - 'ET_s'  : np.ndarray (T, Y, X) – Évaporation du sol.
        - 'ET_i'  : np.ndarray (T, Y, X) – Évaporation de l'interception.
        - 'f_IPAR': np.ndarray (T, Y, X) – Fraction du PAR intercepté.
        - 'f_APAR': np.ndarray (T, Y, X) – Fraction du PAR absorbé.
        - 'f_c'   : np.ndarray (T, Y, X) – Fraction de couverture végétale.
        - 'LAI'   : np.ndarray (T, Y, X) – Indice de surface foliaire.
        - 'R_ns'  : np.ndarray (T, Y, X) – Rayonnement net au sol.
        - 'R_nc'  : np.ndarray (T, Y, X) – Rayonnement net de la canopée.
        - 'T_opt' : np.ndarray (Y, X)    – Température optimale de croissance.
        - 'f_wet' : np.ndarray (T, Y, X) – Contrainte d'humidité de surface.
        - 'f_g'   : np.ndarray (T, Y, X) – Contrainte de conductance verte.
        - 'f_M'   : np.ndarray (T, Y, X) – Contrainte de maturité de la plante.
        - 'f_T'   : np.ndarray (T, Y, X) – Contrainte de température optimale.
        - 'f_SM'  : np.ndarray (T, Y, X) – Contrainte d'humidité du sol.

    Notes
    -----
    - Le terme de Priestley-Taylor  Δ / (Δ + γ)  remplace le calcul
      complet de Penman-Monteith, éliminant le besoin de résistances
      aérodynamiques et stomatiques explicites.
    - La contrainte f_SM utilise la fonction sinus (RH - sin(2πRH)/(2π))
      pour obtenir un comportement non-linéaire réaliste : quasi-nul
      en conditions très sèches, croissance rapide autour de RH ≈ 0.5,
      et saturation progressive vers 1.0 en conditions humides.
    """

    LOGGER.info("🌿 Démarrage du calcul PT-SINRH...")

    # ===================================================================
    # ÉTAPE 1 : Pré-traitement des entrées
    # ===================================================================
    # Borner le NDVI pour éviter les valeurs aberrantes (eau, ombre, etc.)
    NDVI = np.clip(NDVI, 0.05, 1.0)

    # Borner l'humidité relative entre 0 et 1
    RH = np.clip(RH, 0.0, 1.0)

    LOGGER.info(f"   📐 Dimensions des cubes : {NDVI.shape} (temps, y, x)")

    # ===================================================================
    # ÉTAPE 2 : Variables de végétation
    # ===================================================================

    # Fraction du rayonnement photosynthétiquement actif intercepté (fIPAR)
    # Relation linéaire empirique avec le NDVI (Ruimy et al., 1994)
    f_IPAR = M2 * NDVI + B2
    f_IPAR = np.clip(f_IPAR, 1e-6, 0.95)  # Borne physique : ]0, 0.95]

    # Fraction du rayonnement photosynthétiquement actif absorbé (fAPAR)
    # Relation linéaire empirique avec le SAVI (Myneni & Williams, 1994)
    f_APAR = M1 * SAVI + B1

    # Fraction de couverture végétale (fc ≡ fIPAR)
    f_c = f_IPAR.copy()

    # Indice de surface foliaire (LAI) via l'inversion de Beer-Lambert
    # f_c = 1 - exp(-k_PAR * LAI)  ⟹  LAI = -ln(1 - f_c) / k_PAR
    LAI = -np.log(1.0 - f_c) / K_PAR

    LOGGER.info(f"   🌱 LAI moyen = {np.nanmean(LAI):.2f} | fc moyen = {np.nanmean(f_c):.2f}")

    # ===================================================================
    # ÉTAPE 3 : Partition du rayonnement net (sol / canopée)
    # ===================================================================
    # Le rayonnement net est atténué exponentiellement à travers la canopée
    # selon la loi de Beer-Lambert (Campbell & Norman, 1998)
    R_ns = Rn * np.exp(-K_RN * LAI)   # Rayonnement net atteignant le sol
    R_nc = Rn - R_ns                   # Rayonnement net absorbé par la canopée

    LOGGER.info(f"   ☀️ Rn moyen = {np.nanmean(Rn):.1f} W/m² | "
                f"R_nc = {np.nanmean(R_nc):.1f} | R_ns = {np.nanmean(R_ns):.1f}")

    # ===================================================================
    # ÉTAPE 4 : Température optimale de croissance (T_opt)
    # ===================================================================
    # T_opt est la valeur de T_max au pas de temps où la productivité
    # potentielle de la végétation (proxy = PAR * fAPAR * T_max / VPD)
    # atteint son maximum annuel.
    #
    # Le maximum est calculé uniquement sur l'axe temporel (axis=0)
    # pour obtenir une carte spatiale 2D (Y, X) de T_opt.

    # Sécuriser la division par VPD (éviter division par zéro lors de brouillards hivernaux)
    # On limite le VPD à un minimum de 0.2 kPa pour éviter l'explosion de la productivité
    VPD_safe = np.where(VPD > 0.2, VPD, 0.2)

    # Calcul de la productivité. On force à zéro les jours où la végétation est peu active (NDVI < 0.3)
    # pour s'assurer que T_opt correspond bien à un jour de pleine saison de croissance.
    productivity_proxy = np.where(NDVI > 0.3, PAR * f_APAR * T_max / VPD_safe, 0.0)  # (T, Y, X)

    # Indice temporel du maximum de productivité pour chaque pixel
    idx_max = np.nanargmax(productivity_proxy, axis=0)     # (Y, X)

    # Extraire T_max au pas de temps optimal pour chaque pixel (Y, X)
    # Utilisation d'un indexage avancé NumPy sur l'axe temporel
    ny, nx = T_max.shape[1], T_max.shape[2]
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    T_opt = T_max[idx_max, yy, xx]  # (Y, X)

    # lambda_T = T_opt (largeur de la gaussienne thermique)
    lambda_T = T_opt.copy()

    LOGGER.info(f"   🌡️ T_opt moyen = {np.nanmean(T_opt):.1f}°C")

    # ===================================================================
    # ÉTAPE 5 : Contraintes éco-physiologiques (facteurs f)
    # ===================================================================

    # --- f_wet : Contrainte d'humidité de surface (surface mouillée) ---
    # Puissance 4 de RH → très sensible : seules les humidités très
    # élevées (> 0.9) produisent un f_wet significatif (pluie récente).
    f_wet = RH ** 4

    # --- f_g : Contrainte de conductance verte (green) ---
    # Rapport entre la lumière absorbée (fAPAR) et interceptée (fIPAR).
    # Un rapport proche de 1 = feuilles saines ; < 1 = feuilles sénescentes.
    f_IPAR_safe = np.where(f_IPAR > 1e-6, f_IPAR, 1e-6)
    f_g = f_APAR / f_IPAR_safe

    # --- f_M : Contrainte de maturité de la plante ---
    # Compare le fAPAR actuel au fAPAR maximal observé sur la période
    # temporelle pour chaque pixel → indicateur phénologique.
    f_APARmax = np.nanmax(f_APAR, axis=0, keepdims=True)  # (1, Y, X)
    f_APARmax_safe = np.where(f_APARmax > 1e-6, f_APARmax, 1e-6)
    f_M = f_APAR / f_APARmax_safe

    # --- f_T : Contrainte de température optimale ---
    # Gaussienne centrée sur T_opt : la transpiration est maximale quand
    # la température du jour est proche de la température optimale.
    # T_opt est (Y, X), broadcasté automatiquement sur l'axe T.
    lambda_T_safe = np.where(lambda_T > 1e-6, lambda_T, 1e-6)
    f_T = np.exp(-((T_max - T_opt[np.newaxis, :, :]) / lambda_T_safe[np.newaxis, :, :]) ** 2)

    # --- f_SM : Contrainte d'humidité du sol ---
    # Fonction sinus de l'humidité relative :
    #   f_SM = RH - sin(2π·RH) / (2π)
    # Comportement : quasi-nul pour RH < 0.2, croissance sigmoïde,
    # saturation vers 1.0 pour RH > 0.8.
    f_SM = RH - np.sin(2.0 * np.pi * RH) / (2.0 * np.pi)

    LOGGER.info(f"   🔧 f_wet moyen = {np.nanmean(f_wet):.3f} | "
                f"f_SM moyen = {np.nanmean(f_SM):.3f} | "
                f"f_T moyen = {np.nanmean(f_T):.3f}")

    # ===================================================================
    # ÉTAPE 6 : Calcul des trois composantes de l'ET
    # ===================================================================

    # Terme de Priestley-Taylor : Δ / (Δ + γ)
    PT_ratio = Delta / (Delta + GAMMA_PSY)

    # --- ET_c : Transpiration de la canopée ---
    # La canopée transpire proportionnellement à :
    #   - l'énergie disponible dans la canopée (R_nc),
    #   - la conductance verte (f_g), la maturité (f_M),
    #   - la proximité à la température optimale (f_T),
    #   - et uniquement la fraction NON mouillée (1 - f_wet).
    ET_c = (1.0 - f_wet) * ALPHA_PT * f_g * f_M * f_T * PT_ratio * R_nc

    # --- ET_s : Évaporation du sol ---
    # Le sol évapore en fonction de :
    #   - l'énergie disponible au sol (R_ns - G),
    #   - la contrainte d'humidité du sol (f_SM) pour la partie sèche,
    #   - et directement pour la partie mouillée (f_wet).
    ET_s = (f_wet + f_SM * (1.0 - f_wet)) * ALPHA_PT * PT_ratio * (R_ns - G)

    # --- ET_i : Évaporation de l'interception ---
    # L'eau interceptée par les feuilles (rosée, pluie récente) s'évapore
    # proportionnellement à f_wet et à l'énergie de la canopée (R_nc).
    ET_i = f_wet * ALPHA_PT * PT_ratio * R_nc

    # --- ET totale ---
    ET = ET_c + ET_s + ET_i

    # Forcer les valeurs négatives à zéro (physiquement impossible)
    ET   = np.maximum(ET, 0.0)
    ET_c = np.maximum(ET_c, 0.0)
    ET_s = np.maximum(ET_s, 0.0)
    ET_i = np.maximum(ET_i, 0.0)

    LOGGER.info(f"   ✅ ET moyen = {np.nanmean(ET):.1f} W/m² | "
                f"ET_c = {np.nanmean(ET_c):.1f} | "
                f"ET_s = {np.nanmean(ET_s):.1f} | "
                f"ET_i = {np.nanmean(ET_i):.1f}")

    # ===================================================================
    # RETOUR DES RÉSULTATS
    # ===================================================================
    return {
        # Composantes ET
        'ET':    ET,
        'ET_c':  ET_c,
        'ET_s':  ET_s,
        'ET_i':  ET_i,
        # Variables de végétation
        'f_IPAR': f_IPAR,
        'f_APAR': f_APAR,
        'f_c':    f_c,
        'LAI':    LAI,
        # Partition du rayonnement
        'R_ns':   R_ns,
        'R_nc':   R_nc,
        # Température optimale
        'T_opt':  T_opt,
        # Contraintes éco-physiologiques
        'f_wet':  f_wet,
        'f_g':    f_g,
        'f_M':    f_M,
        'f_T':    f_T,
        'f_SM':   f_SM,
    }
