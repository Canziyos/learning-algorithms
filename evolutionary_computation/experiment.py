# tiny one-off check — paste anywhere you can run it
import numpy as np
from extract_data import parse_data
from utils import tour_length  # this now uses EUC_2D per-edge rounding

coords = parse_data("berlin52.tsp")

tour = np.array([0, 34, 38, 39, 37, 36, 33, 35, 48, 31, 44, 18, 40, 7, 8, 9,
                 42, 32, 50, 11, 10, 51, 13, 12, 46, 25, 26, 27, 24, 3, 5, 14,
                 4, 23, 47, 45, 43, 15, 28, 49, 19, 22, 29, 1, 6, 41, 20, 16,
                 2, 17, 30, 21, 0], dtype=int)

# TSPLIB EUC_2D (per-edge rounding) – uses your current utils.tour_length
print("EUC_2D length:", tour_length(tour, coords))

# Continuous (no rounding), for comparison
def tour_length_continuous(tour, data):
    c = data[tour]
    seg = np.sqrt(((c[1:] - c[:-1])**2).sum(axis=1))
    return float(seg.sum())

print("Continuous length:", tour_length_continuous(tour, coords))
