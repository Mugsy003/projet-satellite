import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Traitement.calcul_ET_PT_SINRH import calculate_pt_sinrh_et
import logging
logging.basicConfig(level=logging.INFO)

# Cube synthetique : 5 pas de temps, grille 3x3
T, Y, X = 5, 3, 3
np.random.seed(42)

RH    = np.random.uniform(0.3, 0.9, (T, Y, X))
Rn    = np.random.uniform(200, 500, (T, Y, X))
G     = Rn * 0.1
T_max = np.random.uniform(15, 35, (T, Y, X))
NDVI  = np.random.uniform(0.2, 0.8, (T, Y, X))
SAVI  = NDVI * 0.8
Delta = 0.04 * np.exp(0.06 * T_max)
VPD   = np.random.uniform(0.5, 3.0, (T, Y, X))
PAR   = Rn * 0.48

result = calculate_pt_sinrh_et(RH, Rn, G, T_max, NDVI, SAVI, Delta, VPD, PAR)

print("--- Resultats du test ---")
print("ET shape:   ", result["ET"].shape)
print("T_opt shape:", result["T_opt"].shape)
print("ET moyenne: ", round(np.nanmean(result["ET"]), 2), "W/m2")
print("ET_c moy:   ", round(np.nanmean(result["ET_c"]), 2), "W/m2")
print("ET_s moy:   ", round(np.nanmean(result["ET_s"]), 2), "W/m2")
print("ET_i moy:   ", round(np.nanmean(result["ET_i"]), 2), "W/m2")
print("Toutes valeurs >= 0 :", np.all(result["ET"] >= 0))
print("Test OK!")
