# 🌍 Projet Satellite : Extraction, Désagrégation et Validation LST

Ce projet a pour but de télécharger des données satellitaires multi-capteurs (Landsat 8/9, Sentinel-2, ECOSTRESS), d'en extraire la Température de Surface Terrestre (LST) et des indices optiques, puis d'appliquer des algorithmes de *downscaling spatial* (sharpening) pour améliorer la résolution spatiale de la LST.
Enfin, le projet compare ces températures satellitaires avec des mesures de stations au sol (réseaux ICOS et NOAA).

## 📊 Pipeline de Traitement

Le pipeline est divisé en quatre grandes étapes :
1. **Extraction** : Requêtage des catalogues STAC pour trouver les images sans télécharger de données inutiles.
2. **Transformation** : Téléchargement effectif, masquage des nuages, calcul des indices (NDVI, NDWI, etc.) et de la LST.
3. **Désagrégation (Sharpening / Fusion)** : Amélioration de la résolution spatiale thermique grâce à l'optique.
4. **Validation Ground-Truth** : Comparaison avec les données terrain.

---

## ⚙️ Configuration
Avant de lancer les scripts, vous pouvez paramétrer votre environnement dans le fichier **`config.py`** :
- `SITES_PILOTES` : Coordonnées (lon, lat) des sites (ex: Gebesee, Greece, NOAA SURFRAD...).
- `TIME_OF_INTEREST` : Période temporelle d'extraction (ex: "2022-01-01/2024-08-31").
- `TIME_MARGIN_MINUTES` : Fenêtre temporelle (ex: 30 minutes) pour trouver des couples satellites (ex: ECOSTRESS et Sentinel-2).
- `ltd` : Seuil de couverture nuageuse maximum accepté.

---

## 🚀 Guide d'Utilisation (Commandes)

Exécutez toujours les scripts depuis la racine du projet (là où se trouve ce `README.md`).

### Étape 1 : Extraction (Catalogues STAC)
Ces scripts ne téléchargent pas les images lourdes (GeoTIFF), mais génèrent des "manifestes" (`.json`) listant les identifiants des images valides.

- **Pour Landsat** : 
  ```bash
  python -m Extraction.main_extract
  ```
- **Pour le couple ECOSTRESS / Sentinel-2** (Extraction intelligente croisée) :
  ```bash
  python -m Extraction.main_extract_paires_eco_s2
  ```
  *(Ce script cherche des images S2 sans nuages, puis trouve les images ECOSTRESS correspondantes à ±30 min, optimisant ainsi les téléchargements).*

### Étape 2 : Transformation (Calculs et TIF)
Ces scripts lisent les manifestes, téléchargent les données réelles et calculent les indices (NDVI, SAVI...) ainsi que la LST. Les résultats sont sauvegardés dans le dossier `Outputs/`.

- **Transformation Landsat** (Optique + Thermique 30m) :
  ```bash
  python -m Transform.main_transform
  ```
- **Transformation Sentinel-2** (Indices optiques 10m) :
  ```bash
  python -m Transform.main_transform_sentinel
  ```
- **Transformation ECOSTRESS** (Thermique ~70m) :
  ```bash
  python -m Transform.main_transform_ecostress
  ```

### Étape 3 : Sharpening & Fusion (Désagrégation spatiale)
Ces algorithmes utilisent la relation entre la végétation (indices optiques HD) et la température (basse résolution) pour simuler une LST Haute Résolution.

- **Sharpening Landsat pur (100m -> 30m)** :
  ```bash
  python -m Transform.dms_sharpening
  python -m Transform.tsharp
  ```
- **Fusion ECOSTRESS + Sentinel-2 (~70m -> 10m)** :
  ```bash
  python -m Transform.dms_sharpening_fusion
  python -m Transform.tsharp_fusion
  ```

### Étape 4 : Validation et Performances
Une fois les images générées, on les compare aux mesures des stations locales.

- **Extraction des valeurs pixels et croisement avec ICOS/NOAA** :
  ```bash
  python comparaison_ICOS.py
  ```
  *(Génère un fichier CSV global de comparaison dans `Outputs/` et des séries temporelles)*
  
- **Visualisation des performances (RMSE, MAE, Biais)** :
  ```bash
  python visualisation_performances.py
  ```
  *(Génère des boîtes à moustaches (boxplots) triées avec élimination des outliers)*

---

## 📁 Architecture des Outputs

Tous les résultats sont générés dans le dossier `Outputs/` :
```text
Outputs/
├── manifest_extraction*.json          # Fichiers de pilotage
├── Previews/                          # Aperçus rapides (Quicklooks)
├── Comparaison_ICOS_Global.csv        # Données croisées Satellite/Station
├── Performances_Modeles_RMSE_MAE_Bias_Global.png  # Graphiques de perf
├── Serie_Temporelle_{Site}/           # Données Landsat
│   └── 3_Indices/TIF_Data/            # LST, NDVI, DMS, TsHARP...
├── Serie_Temporelle_{Site}_S2/        # Données Sentinel-2
│   └── 3_Indices/TIF_Data/            # NDVI, NDWI (10m)...
└── Serie_Temporelle_{Site}_ECOSTRESS/ # Données ECOSTRESS
    └── TIF_Data/                      # LST (70m) + LST_Sharpened (10m)
```

## 🧠 Modèles de Machine Learning utilisés
- **DMS (Data Mining Sharpening)** : Utilise un algorithme `RandomForestRegressor` pour apprendre la relation non-linéaire entre une multitude d'indices spatiaux (NDVI, EVI, MNT, coordonnées) et la température. Modèle très robuste.
- **TsHARP** : Utilise une simple régression quadratique (Moindres Carrés) liant uniquement le NDVI à la température. Modèle très rapide, standard de la littérature classique.
