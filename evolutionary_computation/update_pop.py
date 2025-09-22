import numpy as np

from utils import validate_population


def _argsort_desc(fitnesses):
    """Return indices sorting by fitness descending,
    treat NaNs as worst."""
    f = np.nan_to_num(fitnesses, nan=-np.inf)
    return np.argsort(f)[::-1]


def pickup_best(population, k):
    """
    Return the top-k individuals by fitness (descending).
    If k==0, returns empty arrays.
    """
    tours, fitnesses, lengths = population
    N, L = validate_population(population)
    k = int(np.clip(k, 0, N))

    if k == 0:
        return (np.empty((0, L), dtype=tours.dtype),
                np.empty((0,), dtype=fitnesses.dtype),
                np.empty((0,), dtype=lengths.dtype))

    best_idx = _argsort_desc(fitnesses)[:k]
    return tours[best_idx], fitnesses[best_idx], lengths[best_idx]


def update_population(old_population, offspring, n_eliter=1, mode="elitism"):
    """
    Update strategy:
      - 'offspring': replace with offspring; if sizes differ, keep the best N from offspring.
      - 'elitism': always carry top-k elites from OLD; fill remaining N-k primarily from children;
                if children are insufficient, fill the rest from best non-elite OLD.
      - 'bestN': take best N from a union of OLD and CHILDREN.
    Always returns exactly N individuals (matching old_population size).
    """
    # Validate inputs and establish N.
    N_old, L_old = validate_population(old_population)
    N_child, L_child = validate_population(offspring)
    # assert L_old == L_child, "Tour lengths (number of genes) must match between old and offspring."

    tours_old, fits_old, lens_old = old_population
    tours_c, fits_c, lens_c = offspring

    n_eliter = int(np.clip(n_eliter, 0, N_old))

    if mode == "offspring":
        # Replace entirely with offspring;
        # if offspring bigger, keep best N; if smaller, error.
        if N_child >= N_old:
            return pickup_best(offspring, N_old)
        else:
            raise ValueError(f"Offspring size ({N_child}) is smaller than population size ({N_old}) in 'offspring' mode.")

    elif mode == "elitism":
        # 1) Take top-k elites from OLD.
        elites_t, elites_f, elites_l = pickup_best(old_population, n_eliter)

        # 2) check how many slots remain.
        need = N_old - n_eliter
        if need == 0:
            # Degenerate case: population is exactly the elites.
            return elites_t, elites_f, elites_l

        # 3) Primary fill from cgildren: take best 'need' from offspring (or all if fewer).
        take_from_children = min(need, N_child)
        child_best_t, child_best_f, child_best_l = pickup_best(offspring, take_from_children)

        # 4) If still short, fill remainder from best non-elite OLD.
        short = need - take_from_children
        if short > 0:
            # Exclude elite indices from old.
            elite_idx = set(_argsort_desc(fits_old)[:n_eliter])
            all_desc = _argsort_desc(fits_old)

            non_elite_desc = [i for i in all_desc if i not in elite_idx]
            
            if len(non_elite_desc) < short:
                raise ValueError("Not enough candidates to fill the population in 'elitism' mode.")
            filler_idx = np.array(non_elite_desc[:short], dtype=int)
            filler_t = tours_old[filler_idx]
            filler_f = fits_old[filler_idx]
            filler_l = lens_old[filler_idx]

            # Combine elites + best children + fillers.
            new_tours = np.vstack([elites_t, child_best_t, filler_t])
            new_fits  = np.concatenate([elites_f, child_best_f, filler_f])
            new_lens  = np.concatenate([elites_l, child_best_l, filler_l])
        else:
            # Exactly enough from children; combine elites + children.
            new_tours = np.vstack([elites_t, child_best_t])
            new_fits  = np.concatenate([elites_f, child_best_f])
            new_lens  = np.concatenate([elites_l, child_best_l])

        # Sanity: exact N.
        assert new_tours.shape[0] == N_old
        return new_tours, new_fits, new_lens

    elif mode == "bestN":
        # Pool old + children, keep best N.
        tours_pool = np.vstack([tours_old, tours_c])
        fits_pool = np.concatenate([fits_old, fits_c])
        lens_pool = np.concatenate([lens_old, lens_c])
        return pickup_best((tours_pool, fits_pool, lens_pool), N_old)

    else:
        raise ValueError(f"Unknown update mode: {mode}")


def should_stop(generation, evaluations, best_fitness,
                max_generations, max_evaluations, target_fitness=None):
    """
    Stop if:
      - target_fitness reached, or
      - evaluations >= max_evaluations, or
      - generation  >= max_generations.
    """
    if target_fitness is not None and best_fitness >= target_fitness:
        return True
    if evaluations >= max_evaluations:
        return True
    if generation >= max_generations:
        return True
    return False
