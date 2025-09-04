import numpy as np

# Constraint (assignment).
MAX_TOTAL_DIST = 50000

def ecludian_dist(p1, p2):
    """Compute Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)

# Representation.
def build_random_tour(n_cities):
    """Create a random tour visiting all cities once and returning to start."""
    start_loc = 0
    end_loc = 0
    shuffled_indices = np.random.permutation(np.arange(1, n_cities))
    return np.concatenate([[start_loc], shuffled_indices, [end_loc]])

# Decode.
def map_to_coordinates(tour, data):
    """Map a tour (indices) back to coordinates."""
    return data[tour]

def tour_length(tour, data):
    """Compute total length of a tour using vectorized NumPy operations."""
    coords = map_to_coordinates(tour, data)
    diffs = coords[1:] - coords[:-1]       # consecutive differences
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.sum(segment_lengths)

def tour_fitness(tour, data):
    """Compute fitness and length of a tour."""
    length = tour_length(tour, data)
    fitness = 0.0 if length > MAX_TOTAL_DIST else 1.0 / length
    return fitness, length

def init_population(pop_size, data):
    """Initialize a population of tours, fitness, and lengths as NumPy arrays."""
    n_cities = len(data)
    tours = np.zeros((pop_size, n_cities + 1), dtype=np.int32)
    lengths = np.zeros(pop_size, dtype=np.float64)
    fitnesses = np.zeros(pop_size, dtype=np.float64)

    for i in range(pop_size):
        tour = build_random_tour(n_cities)
        fitness, length = tour_fitness(tour, data)
        tours[i] = tour
        fitnesses[i] = fitness
        lengths[i] = length

    return tours, fitnesses, lengths

def rank_probabilities(N, sel_pres):
    probs = np.zeros(N)
    for rank in range(1, N+1):
        p = (sel_pres - (2*sel_pres - 2)*((rank- 1)/(N - 1)))/N
        probs[rank-1] = p
    return probs