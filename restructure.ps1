# Step 1: Create directories
$dirs = @(
    "Extraction/Landsat",
    "Extraction/ECOSTRESS",
    "Extraction/Sentinel2",
    "Extraction/ICOS",
    "Extraction/utils",
    "Transform/Landsat",
    "Transform/ECOSTRESS",
    "Transform/Sentinel2",
    "Transform/Fusion",
    "Transform/common",
    "Analyse",
    "donnees_Gol"
)

foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Force -Path $d | Out-Null
    }
}

# Function to move if source exists
function Move-IfExists {
    param([string]$src, [string]$dest)
    if (Test-Path $src) {
        git mv $src $dest
        if ($LASTEXITCODE -ne 0) {
            # Fallback to normal move if git mv fails (e.g., untracked file)
            Move-Item -Path $src -Destination $dest -Force
        }
    }
}

# Step 2: Move Extraction files
Move-IfExists "Extraction/main_extract.py" "Extraction/Landsat/main_extract_landsat.py"
Move-IfExists "Extraction/stac_client.py" "Extraction/Landsat/stac_client_landsat.py"
Move-IfExists "Extraction/main_extract_ecostress.py" "Extraction/ECOSTRESS/main_extract_ecostress.py"
Move-IfExists "Extraction/stac_client_ecostress.py" "Extraction/ECOSTRESS/stac_client_ecostress.py"
Move-IfExists "Extraction/main_extract_sentinel.py" "Extraction/Sentinel2/main_extract_sentinel.py"
Move-IfExists "Extraction/main_extract_paires_eco_s2.py" "Extraction/Sentinel2/main_extract_paires_eco_s2.py"
Move-IfExists "Extraction/stac_client_sentinel.py" "Extraction/Sentinel2/stac_client_sentinel.py"
Move-IfExists "Extraction/extraction_ICOS.py" "Extraction/ICOS/extraction_ICOS.py"
Move-IfExists "Extraction/extraction_NOAA.py" "Extraction/ICOS/extraction_NOAA.py"
Move-IfExists "Extraction/extract_grece.py" "Extraction/utils/extract_grece.py"
Move-IfExists "Extraction/extract_temperatures.py" "Extraction/utils/extract_temperatures.py"
Move-IfExists "Extraction/generer_dates_existantes.py" "Extraction/utils/generer_dates_existantes.py"

# Step 3: Move Transform files
Move-IfExists "Transform/main_transform.py" "Transform/Landsat/main_transform_landsat.py"
Move-IfExists "Transform/processor.py" "Transform/Landsat/processor_landsat.py"
Move-IfExists "Transform/dms_sharpening.py" "Transform/Landsat/dms_sharpening_landsat.py"
Move-IfExists "Transform/tsharp.py" "Transform/Landsat/tsharp_landsat.py"
Move-IfExists "Transform/main_transform_ecostress.py" "Transform/ECOSTRESS/main_transform_ecostress.py"
Move-IfExists "Transform/processor_ecostress.py" "Transform/ECOSTRESS/processor_ecostress.py"
Move-IfExists "Transform/main_transform_sentinel.py" "Transform/Sentinel2/main_transform_sentinel.py"
Move-IfExists "Transform/processor_sentinel.py" "Transform/Sentinel2/processor_sentinel.py"
Move-IfExists "Transform/dms_sharpening_fusion.py" "Transform/Fusion/dms_sharpening_fusion.py"
Move-IfExists "Transform/tsharp_fusion.py" "Transform/Fusion/tsharp_fusion.py"
Move-IfExists "Transform/indices.py" "Transform/common/indices.py"
Move-IfExists "Transform/fetch_mnt.py" "Transform/common/fetch_mnt.py"
Move-IfExists "Transform/visualizer.py" "Transform/common/visualizer.py"
Move-IfExists "Transform/points.geojson" "Transform/common/points.geojson"

# Step 4: Move Root files to Analyse/
Move-IfExists "analyse.py" "Analyse/analyse.py"
Move-IfExists "calcul_cwsi.py" "Analyse/calcul_cwsi.py"
Move-IfExists "merge_donnees.py" "Analyse/merge_donnees.py"
Move-IfExists "statistiques.py" "Analyse/statistiques.py"

# Step 5: Move CSV files to donnees_Gol/
Move-IfExists "temperatures_Grece.csv" "donnees_Gol/temperatures_Grece.csv"
Move-IfExists "temperatures_Italie.csv" "donnees_Gol/temperatures_Italie.csv"
Move-IfExists "temperatures_Portugal.csv" "donnees_Gol/temperatures_Portugal.csv"
Move-IfExists "donnees_pays.csv" "donnees_Gol/donnees_pays.csv"

# Step 6: Create __init__.py files
foreach ($d in $dirs) {
    if ($d -ne "donnees_Gol") {
        $init_file = "$d\__init__.py"
        if (-not (Test-Path $init_file)) {
            Set-Content -Path $init_file -Value "# Package marker"
        }
    }
}
