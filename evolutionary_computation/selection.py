import numpy as np

#from utils import rank_probabilities
def tournament_s(tours, fitnesses, lengths, n_parents, tournament_size,
                      deterministic=True, p_win=0.8):
    """
    Tournament Selection.
    Returns arrays of shape (n_parents, ...) for (tours, fitnesses, lengths).
    - deterministic=True: best in the sample always wins.
    - deterministic=False: best wins with prob p_win; else pick uniformly from the rest.
    """
    N = len(fitnesses)
    assert N > 0, "Empty population."
    
    k = max(1, min(int(tournament_size), N))
    p_win = float(np.clip(p_win, 0.0, 1.0))

    winners_tours = []
    winner_fitnesses = []
    winners_lengths = []

    for _ in range(n_parents):
        cand_idx = np.random.choice(N, size=k, replace=False)

        if deterministic or k == 1:
            # Best candidate by fitness.
            pick = cand_idx[np.argmax(fitnesses[cand_idx])]
        else:
            # Probabilistic tournament.
            sorted_idx = cand_idx[np.argsort(fitnesses[cand_idx])[::-1]]
            if np.random.rand() < p_win or len(sorted_idx) == 1:
                pick = sorted_idx[0]
            else:
                pick = np.random.choice(sorted_idx[1:])

        # Return COPIES to avoid aliasing parents with population storage.
        winners_tours.append(tours[pick].copy())
        winner_fitnesses.append(fitnesses[pick])
        winners_lengths.append(lengths[pick])

    return (
        np.array(winners_tours, dtype=int),
        np.array(winner_fitnesses, dtype=np.float64),
        np.array(winners_lengths, dtype=np.float64),
    )


def roulette_wheel_s(tours, fitnesses, lengths):
    """
    Roulette Wheel Selection (fitness-proportionate).
    Returns a single (tour, fitness, length).
    """
    N = len(fitnesses)
    assert N > 0, "Empty population."
    total = float(np.sum(fitnesses))

    # If no selection pressure (all zero), pick uniformly at random.
    if total <= 0.0:
        idx = np.random.randint(N)
        return tours[idx], fitnesses[idx], lengths[idx]

    probs = fitnesses / total
    # Numerical safety, proper normalization.
    s = probs.sum()
    if s <= 0:
        idx = np.random.randint(N)
    else:
        probs = probs / s
        idx = np.random.choice(N, p=probs)

    return tours[idx], fitnesses[idx], lengths[idx]

# def lrs(tours, fitnesses, lengths, sel_pres=1.7):
#     """
#     Linear Ranking Selection (single draw).
#     - Rank 0 is best (highest fitness).
#     - Probabilities from rank_probabilities (guards + normalization inside).
#     Returns a single (tour, fitness, length).
#     """
#     N = fitnesses.shape[0]
#     assert N > 0, "Empty population."

#     # Sort indices by fitness DESC so index 0 is the best.
#     indices = np.argsort(fitnesses)[::-1]

#     probs = rank_probabilities(N, sel_pres)
#     # Draw a rank according to ranking probabilities, then map to actual idx.
#     r_selected = np.random.choice(N, p=probs)
#     winner_idx = indices[r_selected]

#     return tours[winner_idx], fitnesses[winner_idx], lengths[winner_idx]
