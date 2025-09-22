# one-off check.
import numpy as np
from extract_data import parse_data
from utils import tour_length, euclidean_dist
p = "berlin52.tsp"
coords = parse_data(path=p)
# print(coords.shape)
coords = coords[0:5]
print(f"coords: {coords}")
# tour = np.array([2, 1, 0])
# mapped = coords[tour]
# print("mapped", mapped)

c = coords[1:] 
d = coords[:-1]
print("C", c)
print("D", d)
# segment_lengths = np.linalg.norm(diffs, axis=1)
# print(segment_lengths)
# print(float(np.sum(segment_lengths)))
# print()




#from utils

# def rank_probabilities(N, sel_pres):
#     """
#     Linear ranking probabilities for ranks 1..N (1 = best).
#     sel_pres is clamped to [1, 2] to ensure a valid distribution.
#     """
#     if N == 1:
#         return np.array([1.0], dtype=np.float64)

#     sel_pres = float(np.clip(sel_pres, 1.0, 2.0))
#     probs = np.zeros(N, dtype=np.float64)
#     for rank in range(1, N + 1):
#         p = (sel_pres - (2 * sel_pres - 2) * ((rank - 1) / (N - 1))) / N
#         probs[rank - 1] = p

#     # Numerical safety.
#     probs = np.maximum(probs, 0.0)
#     s = probs.sum()
#     return probs / s if s > 0 else np.full(N, 1.0 / N, dtype=np.float64)


# def euclidean_dist(p1, p2):
#     """TSPLIB EUC_2D distance: round each edge to nearest integer."""
#     dx = float(p1[0] - p2[0])
#     dy = float(p1[1] - p2[1])
#     return int((dx*dx + dy*dy) ** 0.5 + 0.5)


# def tour_length(tour, data):
#     """Total tour length under TSPLIB EUC_2D (per-edge rounding)."""
#     coords = data[tour]
#     diffs = coords[1:] - coords[:-1]
#     seg = np.sqrt(np.sum(diffs * diffs, axis=1))
#     seg_int = np.floor(seg + 0.5).astype(np.int32)   # round each edge
#     return float(seg_int.sum())


# def euclidean_dist(c1, c2):
#     """Euclidean distance between two points, cities."""
#     return np.linalg.norm(c1 - c2)

lengths = [9000, 7800, 8200, 7500, 7700]
feas_mask = lengths <= 8000   # [False, True, False, True, True]

# np.where(feas_mask)[0] = [1, 3, 4]
idx = np.argmin(lengths[feas_mask])  # argmin over [7800, 7500, 7700] = 1
feas_idx = np.where(feas_mask)[0][idx]
# = [1, 3, 4][1] = 3