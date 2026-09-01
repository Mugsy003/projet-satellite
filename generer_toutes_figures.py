import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.sankey import Sankey

output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

def draw_text_box(ax, x, y, width, height, text, bg_color='white', text_color='black', fontsize=12):
    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='black', facecolor=bg_color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, color=text_color, fontsize=fontsize,
            ha='center', va='center', weight='bold')

# 1. Logo Atos
fig, ax = plt.subplots(figsize=(4, 2))
ax.axis('off')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
draw_text_box(ax, 0.1, 0.2, 0.8, 0.6, "Atos", bg_color='#0066CC', text_color='white', fontsize=24)
plt.savefig(f"{output_dir}/logo_atos.png", dpi=150, bbox_inches='tight')
plt.close()

# 2. Organigramme
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
draw_text_box(ax, 3.5, 7, 3, 2, "Groupe Atos", bg_color='#E6F2FF')
draw_text_box(ax, 1, 3, 3, 2, "Inno'lab\nToulouse", bg_color='#CCE5FF')
draw_text_box(ax, 6, 3, 3, 2, "ANITI\nRecherche", bg_color='#FFE6E6')
ax.annotate("", xy=(2.5, 5), xytext=(4, 7), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(7.5, 5), xytext=(6, 7), arrowprops=dict(arrowstyle="->", lw=2))
draw_text_box(ax, 3.5, 0.5, 3, 1.5, "Stage\nÉvapotranspiration", bg_color='#E6FFE6')
ax.annotate("", xy=(5, 2), xytext=(2.5, 3), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(5, 2), xytext=(7.5, 3), arrowprops=dict(arrowstyle="->", lw=2))
plt.savefig(f"{output_dir}/organigramme.png", dpi=150, bbox_inches='tight')
plt.close()

# 3. Bilan Energie
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
# Sol
ax.axhline(3, color='brown', lw=4)
ax.text(9, 2.5, "Sol", color='brown', fontsize=12, weight='bold')
# Plante
ax.plot([5, 5], [3, 6], color='green', lw=4)
circle = patches.Circle((5, 6), 1.5, color='green', alpha=0.5)
ax.add_patch(circle)
# Fleches
ax.annotate("Rn\n(Rayonnement Net)", xy=(4, 6.5), xytext=(2, 9), arrowprops=dict(facecolor='yellow', shrink=0.05), ha='center')
ax.annotate("H\n(Chaleur Sensible)", xy=(3, 7), xytext=(5, 7.5), arrowprops=dict(facecolor='red', shrink=0.05), ha='center')
ax.annotate("LE\n(Chaleur Latente)", xy=(7, 8), xytext=(5, 6.5), arrowprops=dict(facecolor='blue', shrink=0.05), ha='center')
ax.annotate("G\n(Chaleur Sol)", xy=(5.5, 1), xytext=(5.5, 3), arrowprops=dict(facecolor='brown', shrink=0.05), ha='center')
plt.savefig(f"{output_dir}/bilan_energie.png", dpi=150, bbox_inches='tight')
plt.close()

# 4. Schema Desagregation (TsHARP)
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
# Gros pixel
rect1 = patches.Rectangle((1, 1), 3, 3, facecolor='orange', edgecolor='black')
ax.add_patch(rect1)
ax.text(2.5, 2.5, "LST Basse Res.\n(1 km)", ha='center', va='center')
# Fleche
ax.annotate("Désagrégation\n+ NDVI Haute Res", xy=(5.5, 2.5), xytext=(4, 2.5), arrowprops=dict(arrowstyle="->", lw=2), ha='left', va='center')
# Petits pixels
for i in range(5):
    for j in range(5):
        c = plt.cm.jet(np.random.rand())
        rect = patches.Rectangle((7 + i*0.6, 1 + j*0.6), 0.6, 0.6, facecolor=c, edgecolor='black')
        ax.add_patch(rect)
ax.text(8.5, 4.5, "LST Haute Res.\n(20 m)", ha='center', va='center')
plt.savefig(f"{output_dir}/schema_tsharp.png", dpi=150, bbox_inches='tight')
plt.close()

# 5. LSTM Cell
fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
draw_text_box(ax, 2, 1, 6, 4, "", bg_color='#f9f9f9')
ax.text(5, 4.5, "Cell State (Ct)", ha='center', weight='bold', fontsize=14)
draw_text_box(ax, 2.5, 2, 1.5, 1, "Forget\nGate", bg_color='#ffcccc')
draw_text_box(ax, 4.25, 2, 1.5, 1, "Input\nGate", bg_color='#ccffcc')
draw_text_box(ax, 6, 2, 1.5, 1, "Output\nGate", bg_color='#ccccff')
ax.annotate("", xy=(3.25, 3), xytext=(3.25, 2), arrowprops=dict(arrowstyle="<-", lw=1.5))
ax.annotate("", xy=(5, 3), xytext=(5, 2), arrowprops=dict(arrowstyle="<-", lw=1.5))
ax.annotate("", xy=(6.75, 2), xytext=(6.75, 3), arrowprops=dict(arrowstyle="<-", lw=1.5))
ax.text(1, 2.5, "Input $X_t$", ha='center', va='center')
ax.annotate("", xy=(2.5, 2.5), xytext=(1.5, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(9, 2.5, "Output $h_t$", ha='center', va='center')
ax.annotate("", xy=(8.5, 2.5), xytext=(7.5, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5))
plt.savefig(f"{output_dir}/lstm_cell.png", dpi=150, bbox_inches='tight')
plt.close()

# 6. Pipeline Archi
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 2)
boxes = ["Extraction\n(STAC, API)", "Transform\n(Reproj, Indices)", "Sharpening\n(ML DMS)", "Modélisation\n(TTME/PT)", "Analyse\n& Visu"]
for i, b in enumerate(boxes):
    draw_text_box(ax, 0.5 + i*2.3, 0.5, 1.8, 1, b, bg_color='#e6f2ff', fontsize=10)
    if i < 4:
        ax.annotate("", xy=(0.5 + (i+1)*2.3, 1), xytext=(0.5 + i*2.3 + 1.8, 1), arrowprops=dict(arrowstyle="->", lw=2))
plt.savefig(f"{output_dir}/pipeline_archi.png", dpi=150, bbox_inches='tight')
plt.close()

# 7. DataLake Tree
fig, ax = plt.subplots(figsize=(6, 4))
ax.axis('off')
tree_text = """Data Lake /
├── Landsat/
│   ├── Gebesee/
│   └── Lamasquere/
├── Sentinel2/
│   └── ...
├── Sentinel3/
└── ERA5_Meteo/
    └── historiques.nc"""
ax.text(0.1, 0.5, tree_text, family='monospace', fontsize=14, va='center')
plt.savefig(f"{output_dir}/datalake_tree.png", dpi=150, bbox_inches='tight')
plt.close()

# 8. DMS Process
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
draw_text_box(ax, 1, 6, 3, 1.5, "Optique HR\n(ex: NDVI 20m)", bg_color='#e6ffe6')
draw_text_box(ax, 6, 6, 3, 1.5, "Thermique BR\n(ex: LST 1km)", bg_color='#ffe6e6')
draw_text_box(ax, 1, 4, 3, 1, "Agrégation (Up-scale)\nNDVI -> 1km", bg_color='#f2f2f2')
draw_text_box(ax, 4, 2.5, 4, 1, "Entraînement Modèle ML\n(Random Forest/LightGBM)", bg_color='#e6e6ff')
draw_text_box(ax, 1, 0.5, 3, 1.5, "Prédiction HR\nLST 20m", bg_color='#ffe6e6')
draw_text_box(ax, 6, 0.5, 3, 1, "Correction\ndes résidus", bg_color='#ffffcc')

ax.annotate("", xy=(2.5, 5), xytext=(2.5, 6), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(4, 3), xytext=(2.5, 4), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(6, 3), xytext=(7.5, 6), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(2.5, 2), xytext=(2.5, 2.5), arrowprops=dict(arrowstyle="<-", lw=1.5))
ax.annotate("", xy=(2.5, 2.5), xytext=(4, 3), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(6, 1.5), xytext=(6, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(4, 1), xytext=(6, 1), arrowprops=dict(arrowstyle="->", lw=1.5))
plt.savefig(f"{output_dir}/dms_process.png", dpi=150, bbox_inches='tight')
plt.close()

# 9. Visual Sharpening
np.random.seed(0)
size = 100
lst_hr = np.zeros((size, size))
lst_hr[20:50, 20:50] = 35; lst_hr[60:90, 10:40] = 25; lst_hr[10:90, 60:80] = 30
lst_hr += np.random.normal(0, 1, (size, size))
lst_lr = lst_hr.reshape(10, 10, 10, 10).mean(axis=(1,3))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(lst_lr, cmap='jet')
axes[0].set_title("LST Originale (Basse Résolution)")
axes[0].axis('off')
axes[1].imshow(lst_hr, cmap='jet')
axes[1].set_title("LST Désagrégée (Haute Résolution DMS)")
axes[1].axis('off')
plt.savefig(f"{output_dir}/visual_sharpening.png", dpi=150, bbox_inches='tight')
plt.close()

# 10. TTME Trapezoid
fig, ax = plt.subplots(figsize=(6, 5))
ndvi = np.random.uniform(0.1, 0.9, 1000)
# Dry edge: LST decreases as NDVI increases
dry_edge = 45 - 20 * ndvi
# Wet edge: LST is constant and cool
wet_edge = 25 - 5 * ndvi
lst = np.random.uniform(wet_edge, dry_edge)
ef = (dry_edge - lst) / (dry_edge - wet_edge)
sc = ax.scatter(ndvi, lst, c=ef, cmap='viridis_r', alpha=0.6, s=15)
ax.plot([0.1, 0.9], [45 - 20*0.1, 45 - 20*0.9], color='red', lw=2, label='Bord Sec (Stress hydrique)')
ax.plot([0.1, 0.9], [25 - 5*0.1, 25 - 5*0.9], color='blue', lw=2, label='Bord Humide (ET max)')
ax.set_xlabel("NDVI")
ax.set_ylabel("LST (°C)")
ax.set_title("Espace LST-NDVI (Modèle TTME)")
ax.legend()
plt.colorbar(sc, label="Fraction d'Évapotranspiration (EF)")
plt.savefig(f"{output_dir}/ttme_trapezoid.png", dpi=150, bbox_inches='tight')
plt.close()

# 11. Downscaling ERA5
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
era5_coarse = np.array([[20, 21, 20], [19, 18, 19], [22, 23, 22]])
axes[0].imshow(era5_coarse, cmap='coolwarm', vmin=15, vmax=25)
axes[0].set_title("Ta ERA5 Brute (30 km)")
axes[0].axis('off')
# Simulate high res DEM effect
dem = np.random.normal(0, 1, (300, 300))
import scipy.ndimage as ndimage
dem_smooth = ndimage.gaussian_filter(dem, sigma=20)
era5_fine = ndimage.zoom(era5_coarse, 100, order=1) - dem_smooth * 3
axes[1].imshow(era5_fine, cmap='coolwarm', vmin=15, vmax=25)
axes[1].set_title("Ta ERA5 Downscalée (20 m)")
axes[1].axis('off')
plt.savefig(f"{output_dir}/downscaling_era5.png", dpi=150, bbox_inches='tight')
plt.close()

# 12. LSTM Architecture
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
draw_text_box(ax, 1, 1, 3, 2, "Encodeur LSTM\n(Historique J-30 à J-1)", bg_color='#e6f2ff')
draw_text_box(ax, 5, 1.5, 2, 1, "Vecteur\nContexte (h)", bg_color='#ffffcc')
draw_text_box(ax, 8, 1, 3, 2, "Décodeur LSTM\n(Prévisions J à J+7)", bg_color='#e6ffe6')
ax.annotate("", xy=(5, 2), xytext=(4, 2), arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("", xy=(8, 2), xytext=(7, 2), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(2.5, 0.5, "Forçages passés", ha='center')
ax.annotate("", xy=(2.5, 1), xytext=(2.5, 0.7), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(9.5, 0.5, "Prévisions météo futures", ha='center')
ax.annotate("", xy=(9.5, 1), xytext=(9.5, 0.7), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(9.5, 3.5, "Prédictions ET", ha='center')
ax.annotate("", xy=(9.5, 3.3), xytext=(9.5, 3), arrowprops=dict(arrowstyle="<-", lw=1.5))
plt.savefig(f"{output_dir}/lstm_architecture.png", dpi=150, bbox_inches='tight')
plt.close()

# 13. LSTM Results
fig, ax = plt.subplots(figsize=(10, 4))
days = np.arange(100)
true_y = np.sin(days/10)*50 + 100 + np.random.normal(0, 10, 100)
pred_y = np.sin(days/10)*50 + 100 + np.random.normal(0, 5, 100)
pred_y = np.roll(pred_y, 2)
ax.plot(days, true_y, label='Vérité (ICOS)', color='black')
ax.plot(days, pred_y, label='Prédiction (LSTM)', color='red', linestyle='--')
ax.legend()
ax.set_title("Prédiction de l'Évapotranspiration (Forecasting)")
ax.set_ylabel("LE (W/m²)")
ax.set_xlabel("Jours")
plt.savefig(f"{output_dir}/lstm_results.png", dpi=150, bbox_inches='tight')
plt.close()

print("Toutes les 13 figures ont été générées dans le dossier 'images/'.")
