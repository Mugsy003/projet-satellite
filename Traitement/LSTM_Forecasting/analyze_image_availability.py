import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SITES_PILOTES, OUTPUT_DIR

def extract_datetime_from_filename(filename, is_s2=False):
    base = os.path.basename(filename)
    try:
        date_str = base.split('_')[0]
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception as e:
        return None

def analyze_availability():
    out_dir = os.path.join(OUTPUT_DIR, "Analyses_Graphiques", "Images_Availability")
    os.makedirs(out_dir, exist_ok=True)
    
    all_data = []
    
    for site in SITES_PILOTES:
        landsat_dir = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}", "3_Indices", "TIF_Data")
        s2_dir = os.path.join(OUTPUT_DIR, f"Serie_Temporelle_{site}_S2", "3_Indices", "TIF_Data")
        
        # Landsat
        if os.path.exists(landsat_dir):
            for f in glob.glob(os.path.join(landsat_dir, "*_NDVI.tif")):
                dt = extract_datetime_from_filename(f, is_s2=False)
                if dt:
                    all_data.append({'Site': site, 'Date': dt, 'Satellite': 'Landsat'})
                    
        # Sentinel-2
        if os.path.exists(s2_dir):
            for f in glob.glob(os.path.join(s2_dir, "*_NDVI.tif")):
                dt = extract_datetime_from_filename(f, is_s2=True)
                if dt:
                    all_data.append({'Site': site, 'Date': dt, 'Satellite': 'Sentinel-2'})
                    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("Aucune image trouvée.")
        return
        
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Statistiques
    stats = []
    for site in SITES_PILOTES:
        site_df = df[df['Site'] == site]
        landsat_count = len(site_df[site_df['Satellite'] == 'Landsat'])
        s2_count = len(site_df[site_df['Satellite'] == 'Sentinel-2'])
        total = landsat_count + s2_count
        
        # Gaps analysis
        max_gap = pd.Timedelta(0)
        if total > 1:
            dates_sorted = site_df['Date'].sort_values()
            gaps = dates_sorted.diff().dropna()
            if not gaps.empty:
                max_gap = gaps.max()
                
        stats.append({
            'Site': site,
            'Total Images': total,
            'Landsat': landsat_count,
            'Sentinel-2': s2_count,
            'Plus long trou (jours)': max_gap.days
        })
        
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(os.path.join(out_dir, "statistiques_images.csv"), index=False)
    
    md_table = df_stats.to_markdown(index=False)
    with open(os.path.join(out_dir, "statistiques_images.md"), 'w', encoding='utf-8') as f:
        f.write("# Statistiques de disponibilité des images (2021-2024)\n\n")
        f.write(md_table)
        
    # 2. Graphique chronologique (Timeline)
    plt.figure(figsize=(12, 6))
    
    sites_list = list(reversed(SITES_PILOTES)) # Inverser pour affichage
    
    for i, site in enumerate(sites_list):
        site_df = df[df['Site'] == site]
        
        l_df = site_df[site_df['Satellite'] == 'Landsat']
        s_df = site_df[site_df['Satellite'] == 'Sentinel-2']
        
        plt.scatter(l_df['Date'], [i]*len(l_df), color='blue', marker='|', s=100, alpha=0.7, label='Landsat' if i==0 else "")
        plt.scatter(s_df['Date'], [i]*len(s_df), color='orange', marker='|', s=100, alpha=0.7, label='Sentinel-2' if i==0 else "")
        
    plt.yticks(range(len(sites_list)), sites_list)
    plt.xlabel("Date (Années)")
    plt.title("Disponibilité Temporelle des Images Satellites par Site (2021-2024)")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "chronologie_images.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Analyse terminée. Fichiers sauvegardés dans : {out_dir}")

if __name__ == "__main__":
    analyze_availability()
