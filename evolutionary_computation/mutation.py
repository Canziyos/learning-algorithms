import numpy as np


def mutate_swap(tour, n_swaps=1, prob=0.1):
    assert len(tour) >= 4, "Mutation expects at least 2 interior positions."
    if np.random.rand() >= prob or n_swaps <= 0:
        return tour
    mutated = tour.copy()
    interior = np.arange(1, len(tour) - 1)
    k = min(n_swaps, len(interior) // 2)
    picks = np.random.choice(interior, size=2 * k, replace=False)
    for a, b in zip(picks[::2], picks[1::2]):
        mutated[a], mutated[b] = mutated[b], mutated[a]
    return mutated



def mutate_insert(tour, prob=0.1):
    """
    Remove one interior city and reinsert it at another interior position.
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


def mutate_inverse(tour, prob=0.1):
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
