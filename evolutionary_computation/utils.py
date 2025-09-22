import numpy as np

# Local constraint (temporary). Can be overridden per-call via parameters.
MAX_TOTAL_DIST = 8000

def tour_length(tour, dataset):
    """The total length of a tour."""
    coords = map_to_coordinates(tour, dataset)

    # pick up all city coordinates except for the first,
    # => [c2, c3, .., c52].
    # and another copy of cities from all rows, exclude the last one,
    # => [c1, c2, ...c51].
    # then subtrack them row-wise => (d1 = c2-c1, d2 = c3-c2, ..., d52 = c52-c51] 
    diffs = coords[1:] - coords[:-1]

    # calculate sqrt(d1^2), sqrt(d2^2), ... 
    segment_lengths = np.linalg.norm(diffs, axis=1)
    # sum distances and return with it.
    return float(np.sum(segment_lengths))

# Representation.
def build_random_tour(n_cities):
    """Create a random tour visiting all cities once and returning to start."""
    assert n_cities >= 2, "n_cities must be >= 2 for a closed tour."

    start_loc = 0
    end_loc = 0

    # Generate a random permutation of all cities except 0.
    # Each city appears exactly once.
    shuffled_indices = np.random.permutation(np.arange(1, n_cities))
    return np.concatenate([[start_loc], shuffled_indices, [end_loc]])


# Decode.
def map_to_coordinates(tour, dataset):
    """Map a tour (indices) back to coordinates."""
    return dataset[tour]

def tour_fitness(tour, dataset, max_total_dist=None):
    """The fitness and the length of a tour."""
    threshold = MAX_TOTAL_DIST if max_total_dist is None else max_total_dist
    length = tour_length(tour, dataset)
    fitness = 1.0 / length
    return fitness, length


def init_population(pop_size, dataset, max_total_dist=None):
    """Initialize a population of tours, fitness, and lengths (numpy arrays)."""
    n_cities = len(dataset)
    tours = np.zeros((pop_size, n_cities + 1), dtype=np.int32)
    lengths = np.zeros(pop_size, dtype=np.float64)
    fitnesses = np.zeros(pop_size, dtype=np.float64)

    for i in range(pop_size):
        tour = build_random_tour(n_cities)
        fitness, length = tour_fitness(tour, dataset, max_total_dist=max_total_dist)
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

def validate_population(pop):
    tours, fitnesses, lengths = pop
    assert isinstance(tours, np.ndarray) and isinstance(fitnesses, np.ndarray) and isinstance(lengths, np.ndarray), \
        "Population components must be NumPy arrays."
    
    N = tours.shape[0]
    assert fitnesses.shape == (N,) and lengths.shape == (N,), \
        "Shapes must be: tours (N, L), fitnesses (N,), lengths (N,)."
    
    return N, tours.shape[1]

import json

def deduplicate_survivors(path_in, path_out):
    """Load survivors.json, remove duplicate tours, save to path_out."""
    with open(path_in, "r", encoding="utf-8") as f:
        records = json.load(f)

    seen = set()
    unique_records = []
    for rec in records:
        key = tuple(rec["tour"])
        if key not in seen:
            seen.add(key)
            unique_records.append(rec)

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(unique_records, f, indent=2)

    print(f"Deduplicated: {len(records)} => {len(unique_records)} saved to {path_out}")
