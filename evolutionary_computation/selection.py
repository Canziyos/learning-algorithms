import numpy as np
from utils import rank_probabilities

def rws(tours, fitnesses, lengths):
    """
    Select one individual index using roulette wheel selection.
    Returns (tour, fitness, length).
    """
    total_fitness = np.sum(fitnesses)

    # Edge case: if all fitness are 0, choose randomly
    if total_fitness == 0:
        idx = np.random.randint(len(fitnesses))
        return tours[idx], fitnesses[idx], lengths[idx]

    # Normalize fitness to probabilities
    probabilities = fitnesses / total_fitness

    # Spin the wheel
    idx = np.random.choice(len(fitnesses), p=probabilities)
    return tours[idx], fitnesses[idx], lengths[idx]


def ts(tours, fitnesses, lengths, n_parents, n_candidates, deterministic=True, p_win=0.8):
    """
    Select n_parents winners via tournament selection.
    Returns lists of (tours, fitnesses, lengths) for winners.
    """
    winners_tours = []
    winners_fitnesses = []
    winners_lengths = []

    for _ in range(n_parents):
        # Sample candidate indices
        candidate_indices = np.random.choice(len(fitnesses), size=n_candidates, replace=False)

        if deterministic:
            # Pick the best candidate (max fitness)
            best_idx = candidate_indices[np.argmax(fitnesses[candidate_indices])]
        else:
            # Probabilistic: best wins with probability p_win
            sorted_indices = candidate_indices[np.argsort(fitnesses[candidate_indices])[::-1]]
            if np.random.rand() < p_win:
                best_idx = sorted_indices[0]
            else:
                best_idx = np.random.choice(sorted_indices[1:])

        winners_tours.append(tours[best_idx])
        winners_fitnesses.append(fitnesses[best_idx])
        winners_lengths.append(lengths[best_idx])

    return (np.array(winners_tours),
            np.array(winners_fitnesses),
            np.array(winners_lengths))

def lrs(tours, fitnesses, lengths, sel_pres=1.7):
    N = fitnesses.shape[0] #len(fitnesses)
    indices = np.argsort(fitnesses)
    probs = rank_probabilities(N, sel_pres)
    r_selected = np.random.choice(N, p=probs, replace=False)
    winner_idx = indices[r_selected]

    return tours[winner_idx], fitnesses[winner_idx], lengths[winner_idx]
   
