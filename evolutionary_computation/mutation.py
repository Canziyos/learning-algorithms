import numpy as np


def mutate_swap(tour, n_swaps=1, prob=0.1):
    """
    Swap-mutation on interior cities (excludes start/end depot 0).
    Copy-on-mutate: returns a new array only if mutation happens; otherwise returns the original.
    """
    assert len(tour) >= 4, "Mutation expects at least 2 interior positions."
    if np.random.rand() < prob:
        mutated = tour.copy()
        for _ in range(n_swaps):
            i, j = np.random.choice(range(1, len(tour) - 1), 2, replace=False)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    return tour


def mutate_insertion(tour, prob=0.1):
    """
    Insertion-mutation: remove one interior city and reinsert it at another interior position.
    Copy-on-mutate semantics.
    """
    assert len(tour) >= 4, "Mutation expects at least 2 interior positions."
    if np.random.rand() < prob:
        i, j = np.random.choice(range(1, len(tour) - 1), 2, replace=False)
        src = tour.copy()
        city = src[i]
        tmp = np.delete(src, i)
        if j >= i:
            j -= 1  # adjust because array shrank after deletion
        mutated = np.insert(tmp, j, city)
        return mutated
    return tour


def mutate_inversion(tour, prob=0.1):
    """
    Inversion-mutation: reverse a random interior segment.
    Copy-on-mutate semantics.
    """
    assert len(tour) >= 4, "Mutation expects at least 2 interior positions."
    if np.random.rand() < prob:
        i, j = np.random.choice(range(1, len(tour) - 1), 2, replace=False)
        if i > j:
            i, j = j, i
        mutated = tour.copy()
        mutated[i:j + 1] = mutated[i:j + 1][::-1]
        return mutated
    return tour
