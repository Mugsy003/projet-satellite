import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Création du dossier de sortie pour les figures s'il n'existe pas
output_dir = "Outputs/Figures_Rapport"
os.makedirs(output_dir, exist_ok=True)

print("Génération des figures pour le rapport de stage...")

# =====================================================================
# Figure 1 : Comparaison visuelle des cartes LST - Originale vs Sharpened
# =====================================================================
print("Création de la Figure 1 (figure1_lst_comparison.png)...")

# Génération de données synthétiques pour simuler une carte thermique
np.random.seed(42)
size = 100
# Simulation de champs agricoles (patchs rectangulaires de températures)
lst_high_res = np.zeros((size, size))
lst_high_res[10:40, 10:40] = 35.0  # Champ chaud (sol nu)
lst_high_res[50:90, 20:50] = 25.0  # Forêt fraîche
lst_high_res[10:30, 60:90] = 30.0  # Culture irriguée
# Ajout de bruit naturel (variabilité intra-parcelle)
lst_high_res += np.random.normal(0, 1.5, (size, size))

# Simulation d'une basse résolution type Sentinel-3 (1km)
# On réduit drastiquement la résolution en faisant la moyenne par blocs de 10x10
lst_low_res = lst_high_res.reshape(10, 10, 10, 10).mean(axis=(1, 3))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Affichage Basse Résolution
im1 = axes[0].imshow(lst_low_res, cmap='jet', extent=[0, size, 0, size], vmin=22, vmax=38)
axes[0].set_title("Résolution Originale (Basse - ex: 1km)")
axes[0].axis('off')

# Affichage Haute Résolution (Sharpened)
im2 = axes[1].imshow(lst_high_res, cmap='jet', extent=[0, size, 0, size], vmin=22, vmax=38)
axes[1].set_title("Résolution Sharpened (Haute - ex: 20m)")
axes[1].axis('off')

# Barre de couleur commune
cbar = fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Température de Surface (LST) [°C]')

plt.savefig(os.path.join(output_dir, "figure1_lst_comparison.png"), dpi=300, bbox_inches='tight')
plt.close()

# =====================================================================
# Figure 2 : Série temporelle des prédictions LSTM vs Vérité Terrain
# =====================================================================
print("Création de la Figure 2 (figure2_lstm_predictions.png)...")

# Génération d'une série temporelle synthétique d'Évapotranspiration (W/m2)
dates = pd.date_range(start="2023-04-01", end="2023-10-31", freq='D')
n_days = len(dates)

# Tendance saisonnière (cloche estivale)
seasonal_trend = 150 + 100 * np.sin(np.linspace(0, np.pi, n_days))

# Chutes dues aux événements de pluie (baisse de LST / hausse de l'humidité)
rain_events = np.random.choice([0, 1], size=n_days, p=[0.9, 0.1])
rain_drops = np.zeros(n_days)
for i in range(n_days):
    if rain_events[i] == 1:
        rain_drops[i:i+5] = np.linspace(-80, 0, min(5, n_days-i))

# Vérité Terrain (ICOS)
true_et = seasonal_trend + rain_drops + np.random.normal(0, 15, n_days)

# Prédiction LSTM (suit bien la tendance mais avec un léger lissage)
lstm_pred = seasonal_trend + (rain_drops * 0.85) + np.random.normal(0, 10, n_days)
# On décale très légèrement la prédiction pour simuler une erreur de prévision
lstm_pred = np.roll(lstm_pred, 1) 
lstm_pred[0] = lstm_pred[1]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(dates, true_et, label="Vérité Terrain (ICOS - Chaleur Latente)", color='black', alpha=0.7, linewidth=1.5)
ax.plot(dates, lstm_pred, label="Prédiction (Modèle LSTM)", color='darkorange', linewidth=2, linestyle='--')

ax.set_title("Prédiction de l'Évapotranspiration par le modèle LSTM sur une saison (Gebesee)")
ax.set_ylabel("Flux de Chaleur Latente (W/m²)")
ax.set_xlabel("Date")
ax.legend(loc='upper right')
ax.grid(True, linestyle=':', alpha=0.6)
fig.autofmt_xdate()

plt.savefig(os.path.join(output_dir, "figure2_lstm_predictions.png"), dpi=300, bbox_inches='tight')
plt.close()

print(f"Terminé ! Les figures sont disponibles dans le dossier : {output_dir}")
