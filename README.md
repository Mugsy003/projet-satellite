# 🌍 Projet Satellite : Extraction, Désagrégation et Validation LST

Ce projet a pour but de télécharger des données satellitaires multi-capteurs (Landsat 8/9, Sentinel-2, Sentinel-3, ECOSTRESS), d'en extraire la Température de Surface Terrestre (LST) et des indices optiques, puis d'appliquer des algorithmes de *downscaling spatial* (sharpening) pour améliorer la résolution spatiale de la LST.
Enfin, le projet compare ces températures satellitaires avec des mesures de stations au sol (réseaux ICOS et NOAA), et modélise l'**Évapotranspiration (ET)** de la végétation en s'appuyant sur le modèle TTME (Two-source Trapezoid Model for Evapotranspiration) couplé à des données météorologiques (ERA5 ou In-Situ).

## 🚀 Le Pipeline Automatisé (`main.py`)

L'intégralité du projet peut désormais être exécutée depuis un seul point d'entrée : **`main.py`**.
Ce script orchestrateur lance automatiquement (et dans l'ordre) l'extraction, la transformation, le machine learning, la comparaison avec le terrain et la visualisation.

```bash
python main.py
```

Vous pouvez **activer ou désactiver** chaque étape individuellement en modifiant le dictionnaire `PIPELINE_STEPS` dans le fichier **`config.py`**.

---

## ⚙️ Configuration (`config.py`)

Avant de lancer le projet, paramétrez votre environnement dans **`config.py`** :
- `PIPELINE_STEPS` : Active/désactive les étapes de `main.py` (ex: `"extraction_landsat": True`).
- `SITES_PILOTES` : Coordonnées (lon, lat) des sites terrestres à étudier.
- `TIME_OF_INTEREST` : Période temporelle globale d'extraction.
- `DATE_DEBUT_VISU` / `DATE_FIN_VISU` : Permet de restreindre les graphiques finaux à une saison précise.
- `TIME_MARGIN_MINUTES` : Fenêtre temporelle (ex: 30 minutes) pour trouver des couples satellites parfaits (ex: ECOSTRESS et Sentinel-2).

---

## 📁 Architecture du Projet (Par Mission)

Le code est structuré de façon modulaire, séparé par mission spatiale :

```text
projet-satellite/
├── config.py                         # Fichier de configuration central
├── main.py                           # Orchestrateur global
├── comparaison_ICOS.py               # Validation des pixels LST avec les stations
├── calcul_ET.py                      # 💧 Modélisation de l'Évapotranspiration (TTME)
├── visualisation_performances.py     # Création des graphiques (boxplots)
├── optimisation_pipeline.py          # Tuning des hyperparamètres via Optuna
│
├── Extraction/                       # Recherche STAC et Manifestes
│   ├── Landsat/
│   ├── ECOSTRESS/
│   ├── Sentinel2/
│   ├── Sentinel3/                    # NOUVEAU: Extraction NetCDF (LST SLSTR & Optique Synergy)
│   ├── ICOS/                         # Téléchargement des données météo in-situ (Reference)
│   ├── ERA5/                         # Téléchargement des forçages météo ERA5 (Copernicus)
│   └── utils/
│
├── Transform/                        # Calcul des Indices, LST et Sharpening
│   ├── Landsat/                      # Sharpening intra-capteur (100m -> 30m)
│   ├── ECOSTRESS/
│   ├── Sentinel2/
│   ├── Sentinel3/                    # Sharpening intra-capteur (1km -> 300m)
│   ├── Fusion/                       # Fusion multi-capteur (ECOSTRESS+S2, S3+S2)
│   └── common/
│
├── Analyse/                          # Scripts de statistiques et études annexes
└── donnees_Gol/                      # Fichiers CSV de températures externes
```

---

## 🧠 Modèles de Machine Learning et Optimisation

### Les Algorithmes de Sharpening
- **DMS (Data Mining Sharpening)** : Utilise un algorithme `RandomForestRegressor` ou `LightGBM` pour apprendre la relation complexe et non-linéaire entre une multitude d'indices spatiaux (NDVI, NDWI, SAVI, EVI, MNT, Coordonnées XY) et la température.
- **TsHARP** : Algorithme classique de la littérature basé sur une régression des Moindres Carrés liant uniquement le NDVI à la température.

### Auto-Tuning (Optuna)
Pour trouver les paramètres parfaits du DMS (RandomForest vs LightGBM, profondeur des arbres, etc.), vous pouvez lancer la boucle d'optimisation intelligente. Elle testera des dizaines de configurations et conservera celle qui offre la meilleure **RMSE Terrain** :
```bash
python optimisation_pipeline.py
```

---

## 💧 Modélisation de l'Évapotranspiration (ET)

Le projet inclut désormais un moteur de modélisation de l'Évapotranspiration à l'échelle du pixel, via l'algorithme **TTME** (Two-source Trapezoid Model for Evapotranspiration).

Ce modèle sépare l'évaporation du sol (Soil) de la transpiration des plantes (Canopy) en construisant un espace théorique LST-NDVI (Trapèze) contraint par la thermodynamique (Conservation de l'énergie de rayonnement $R_n$).

### Comment lancer le calcul de l'ET ?
Vous pouvez calculer l'ET en utilisant soit les données météo parfaites du sol (ICOS), soit les données météo spatiales (ERA5) :
```bash
# Calcul avec la météo In-Situ
python calcul_ET.py --source icos

# Calcul avec la météo globale Copernicus (ERA5)
python calcul_ET.py --source era5
```
Les scripts dans le dossier `Analyse/` (comme `generer_graphes_par_site.py`) permettent ensuite de générer des séries temporelles croisant les résultats satellitaires avec les "Ground Truth" des tours à flux.

---

## 📊 Outputs Générés

Tous les résultats sont générés dans le dossier `Outputs/` :
- `Outputs/Validation_Saisonniere_LST.csv` : Bilan global croisant la LST satellite et le terrain.
- `Outputs/Resultats_ET_TTME.csv` : Résultats des calculs d'Évapotranspiration (Flux de chaleur, ET mm/h).
- `Outputs_performances/` : Graphiques finaux d'erreur LST (RMSE, MAE, Boxplots).
- `Outputs/comparaisons ET/` : Graphiques de suivi temporel de l'Évapotranspiration par site.
- `Outputs/Serie_Temporelle_{Site}_[Mission]/` : Les images TIF générées (LST, NDVI, ET_map...).
