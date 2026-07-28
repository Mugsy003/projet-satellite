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
├── Traitement/                       # Modèles physiques et Optimisation
│   ├── TTME/                         # 💧 Modélisation de l'Évapotranspiration (TTME)
│   │   └── run_TTME.py               
│   ├── PT_SINRH/                     # 💧 Modélisation de l'Évapotranspiration (PT-SINRH)
│   │   ├── run_PT_SINRH.py           
│   │   └── algorithme_PT_SINRH.py    
│   ├── downscaling_Ta.py             # Downscaling de la température de l'air (ERA5)
│   └── optimisation_pipeline.py      # Tuning des hyperparamètres via Optuna
│
├── Analyse/                          # Scripts de statistiques et études annexes
│   ├── TTME/                         # Évaluation et graphiques pour TTME
│   │   ├── eval_modele_vs_modele.py
│   │   ├── eval_vs_vrai_ICOS.py
│   │   └── generer_cartes_spatiales.py
│   ├── PT_SINRH/                     # Évaluation et graphiques pour PT-SINRH
│   │   ├── eval_modele_vs_modele.py
│   │   ├── eval_vs_vrai_ICOS.py
│   │   └── generer_graphes.py
│   └── comparaison_ICOS.py           # Validation des pixels LST avec les stations
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
python Traitement/optimisation_pipeline.py
```

---

## 💧 Modélisation de l'Évapotranspiration (ET)

Le projet inclut désormais deux moteurs de modélisation de l'Évapotranspiration à l'échelle du pixel :
1. **TTME (Two-source Trapezoid Model for Evapotranspiration)** : Sépare l'évaporation du sol et la transpiration des plantes via un espace théorique LST-NDVI thermodynamique.
2. **PT-SINRH** : Modèle basé sur Priestley-Taylor et des contraintes éco-physiologiques (humidité, rayonnement, phénologie).

### Comment lancer les calculs ?
Vous pouvez calculer l'ET en utilisant la météo spatiale (ERA5), la météo in-situ (ICOS), ou la météo downscalée (ERA5_DS) :

```bash
# Pour TTME (ex: avec ERA5)
python -m Traitement.TTME.run_TTME --source era5

# Pour PT-SINRH (ex: avec ERA5 Downscalé)
python -m Traitement.PT_SINRH.run_PT_SINRH --source era5_ds
```

Les scripts d'analyse (dans `Analyse/TTME/` et `Analyse/PT_SINRH/`) génèrent ensuite les performances statistiques (RMSE, Biais) et les courbes temporelles contre les données absolues (Chaleur Latente) mesurées par les tours à flux.

---

## 📊 Outputs Générés

Tous les résultats sont générés dans le dossier `Outputs/` :
- `Outputs/Resultats_CSV/` : Tableaux statistiques croisés (LST satellite vs terrain, ET modélisée vs ICOS).
- `Outputs/Analyses_Graphiques/` : Graphiques finaux d'erreur LST et courbes temporelles d'ET.
- `Outputs/Outputs_ICOS/` : Extractions des données de chaleur latente des tours à flux.
- `Outputs/Serie_Temporelle_{Site}/` : Les images TIF générées (LST, NDVI, SAVI, ET...).
