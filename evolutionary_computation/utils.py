import numpy as np

# Local constraint (temporary). Can be overridden per-call via parameters.
MAX_TOTAL_DIST = 8000


# def euclidean_dist(p1, p2):
#     """Compute Euclidean distance between two points."""
#     return np.linalg.norm(p1 - p2)

# def tour_length(tour, data):
#     """Compute total length of a tour."""
#     coords = map_to_coordinates(tour, data)
#     diffs = coords[1:] - coords[:-1]
#     segment_lengths = np.linalg.norm(diffs, axis=1)
#     return float(np.sum(segment_lengths))

def euclidean_dist(p1, p2):
    """TSPLIB EUC_2D distance: round each edge to nearest integer."""
    dx = float(p1[0] - p2[0])
    dy = float(p1[1] - p2[1])
    return int((dx*dx + dy*dy) ** 0.5 + 0.5)


def tour_length(tour, data):
    """Total tour length under TSPLIB EUC_2D (per-edge rounding)."""
    coords = data[tour]
    diffs = coords[1:] - coords[:-1]
    seg = np.sqrt(np.sum(diffs * diffs, axis=1))
    seg_int = np.floor(seg + 0.5).astype(np.int32)   # round each edge
    return float(seg_int.sum())


# Representation.
def build_random_tour(n_cities):
    """Create a random tour visiting all cities once and returning to start."""
    assert n_cities >= 2, "n_cities must be >= 2 for a closed tour."
    start_loc = 0
    end_loc = 0
    shuffled_indices = np.random.permutation(np.arange(1, n_cities))
    return np.concatenate([[start_loc], shuffled_indices, [end_loc]])


# Decode.
def map_to_coordinates(tour, data):
    """Map a tour (indices) back to coordinates."""
    return data[tour]


def tour_fitness(tour, data, max_total_dist=None):
    """Compute fitness and length of a tour."""
    threshold = MAX_TOTAL_DIST if max_total_dist is None else max_total_dist
    length = tour_length(tour, data)
    fitness = 1.0 / length
    return fitness, length


def init_population(pop_size, data, max_total_dist=None):
    """Initialize a population of tours, fitness, and lengths as NumPy arrays."""
    n_cities = len(data)
    tours = np.zeros((pop_size, n_cities + 1), dtype=np.int32)
    lengths = np.zeros(pop_size, dtype=np.float64)
    fitnesses = np.zeros(pop_size, dtype=np.float64)

    for i in range(pop_size):
        tour = build_random_tour(n_cities)
        fitness, length = tour_fitness(tour, data, max_total_dist=max_total_dist)
        tours[i] = tour
        fitnesses[i] = fitness
        lengths[i] = length

    return tours, fitnesses, lengths


def init_n_population(n, pop_size, data, max_total_dist=None):
    """Create n independent populations."""
    populations = []
    for _ in range(n):
        pop = init_population(pop_size, data, max_total_dist=max_total_dist)
        populations.append(pop)
    return populations


def rank_probabilities(N, sel_pres):
    """
    Linear ranking probabilities for ranks 1..N (1 = best).
    sel_pres is clamped to [1, 2] to ensure a valid distribution.
    """
    if N == 1:
        return np.array([1.0], dtype=np.float64)

    sel_pres = float(np.clip(sel_pres, 1.0, 2.0))
    probs = np.zeros(N, dtype=np.float64)
    for rank in range(1, N + 1):
        p = (sel_pres - (2 * sel_pres - 2) * ((rank - 1) / (N - 1))) / N
        probs[rank - 1] = p

    # Numerical safety.
    probs = np.maximum(probs, 0.0)
    s = probs.sum()
    return probs / s if s > 0 else np.full(N, 1.0 / N, dtype=np.float64)
