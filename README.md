# Satellite Time-Series Analysis Pipeline

A comprehensive Landsat time-series analysis pipeline for environmental monitoring across European pilot sites. This project automates the extraction, processing, and analysis of satellite imagery from the Microsoft Planetary Computer STAC catalog.

## 🌍 Project Overview

This pipeline processes Landsat-8/9 satellite data to monitor vegetation health, water presence, urban development, and land surface temperature across 5 European pilot sites: Portugal, Spain, Greece, Italy, and France. The system operates in three main phases: **Extraction**, **Transformation**, and **Statistics**.

### Key Features

- **Automated STAC Data Retrieval**: Connects to Microsoft Planetary Computer to fetch Landsat imagery
- **Intelligent Cloud Filtering**: Applies cloud cover thresholds and QA-based masking
- **Time-Series Processing**: Handles temporal gaps with intelligent median filtering and neighbor searching
- **Spectral Index Calculation**: Computes NDVI, NDWI, NDBI, EVI, SAVI, and LST indices
- **Geospatial Visualization**: Generates RGB compositions and index maps as GeoTIFFs and PNGs
- **Statistical Analysis**: Creates temporal cloud cover analysis with trend visualization
- **Multi-Site Comparison**: Processes data across 5 European locations simultaneously

## 📋 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Output Structure](#output-structure)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Virtual environment support

### Setup Steps

1. Clone or navigate to the project directory:
   ```bash
   cd c:\Users\a951444\Workspace\projet-satellite
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

The pipeline runs in three sequential phases. Execute them in order:

### Phase 1: Data Extraction
Extract satellite imagery from the STAC catalog and generate manifests.

```bash
python -m Extraction.main_extract
```

**What it does:**
- Connects to Microsoft Planetary Computer STAC catalog
- Searches for Landsat-8/9 imagery within 3km radius of each pilot site
- Filters images by cloud cover (<70% threshold)
- Downloads preview images of the clearest scenes
- Creates two manifests: one for all images (statistics) and one for filtered images (processing)

### Phase 2: Data Transformation
Process time-series data and calculate spectral indices.

```bash
python -m Transform.main_transform
```

**What it does:**
- Loads data cubes from STAC items using ODC-STAC
- Applies temporal gap-filling and cloud masking
- Calculates spectral indices (NDVI, NDWI, NDBI, EVI, SAVI, LST)
- Generates RGB compositions and index visualization maps
- Saves processed data as GeoTIFFs and PNGs

### Phase 3: Statistical Analysis
Generate cloud cover statistics and temporal visualizations.

```bash
python statistiques.py
```

**What it does:**
- Analyzes cloud cover patterns from STAC metadata
- Creates time-series plots with moving averages
- Generates histograms showing cloud cover distribution
- Produces comparative visualizations across all sites

## 📁 Project Structure

```
projet-satellite/
├── config.py                    # Global configuration and constants
├── requirements.txt             # Python dependencies
├── statistiques.py              # Cloud statistics analysis
├── antigravity.config.json      # Additional configuration
├── Extraction/                  # Phase 1: Data extraction
│   ├── main_extract.py          # Main extraction script
│   └── stac_client.py           # STAC catalog client
├── Transform/                   # Phase 2: Data processing
│   ├── main_transform.py        # Main transformation script
│   ├── processor.py             # Time-series processing pipeline
│   ├── indices.py               # Spectral index calculations
│   ├── visualizer.py            # Visualization and output generation
│   ├── dms_sharpening.py        # Image sharpening utilities
│   └── fetch_mnt.py             # DEM/MNT data fetching
├── Utils/                       # Utility modules
│   ├── __init__.py
│   ├── geo.py                   # Geospatial utilities
│   ├── image.py                 # Image processing functions
│   ├── ml.py                    # Machine learning utilities
│   ├── stats.py                 # Statistical functions
│   ├── ui.py                    # User interface helpers
│   ├── utils.py                 # General utilities
│   └── vis.py                   # Visualization helpers
├── Outputs/                     # Generated data and results
│   ├── manifest_extraction.json # Filtered image IDs
│   ├── manifest_stats_global.json # All image IDs
│   ├── Statistiques_Nuages/     # Cloud statistics plots
│   └── Serie_Temporelle_*/      # Time-series data per country
│       ├── 1_Brutes/            # Raw data
│       ├── 2_Traitees/          # Processed data
│       └── 3_Indices/           # Spectral indices
└── Previews_Landsat/            # Preview images
    ├── France/
    ├── Greece/
    ├── Italy/
    ├── Portugal/
    └── Spain/
```

## ⚙️ Configuration

Key parameters are defined in `config.py`:

- **Analysis Period**: `TIME_OF_INTEREST = "2024-09-15/2025-09-15"`
- **Spectral Bands**: NIR, Red, Green, Blue, QA Pixel, LWIR, SWIR
- **Cloud Threshold**: `lt = 70` (70% maximum cloud cover)
- **Search Radius**: `radius_km = 3` km around each pilot site
- **Pilot Sites**: GPS coordinates for 5 European locations

Modify these values in `config.py` to customize the analysis.

## 📦 Dependencies

The project requires the following Python packages:

- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `requests` - HTTP client
- `pystac-client` - STAC protocol client
- `planetary-computer` - Microsoft Planetary Computer access
- `odc-stac` - STAC data loading into xarray
- `scikit-learn` - Machine learning algorithms
- `scipy` - Scientific computing
- `xarray` - Multi-dimensional arrays
- `pystac` - STAC library
- `scikit-image` - Image processing
- `cubist` - Modeling framework
- `rioxarray` - Geospatial raster operations
- `odc` - Open Data Cube utilities
- `pandas` - Data manipulation

## 📊 Output Structure

The pipeline generates organized outputs in the `Outputs/` directory:

- **Manifests**: JSON files containing STAC item IDs for processing
- **Time-Series Data**: Organized by country and processing level
  - Raw data (1_Brutes)
  - Processed data (2_Traitees)
  - Spectral indices (3_Indices)
- **Visualizations**: RGB compositions, index maps, and statistical plots
- **Statistics**: Cloud cover analysis and temporal trends

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed for environmental monitoring research. Please check with the project maintainers for usage permissions.

## 🆘 Troubleshooting

- **STAC Connection Issues**: Ensure internet connectivity and valid Planetary Computer access
- **Memory Errors**: Reduce the analysis period or increase system RAM
- **Missing Dependencies**: Run `pip install -r requirements.txt` in the activated virtual environment
- **File Path Issues**: Ensure the working directory is set to the project root

For additional support, check the configuration files and utility modules for detailed logging information.