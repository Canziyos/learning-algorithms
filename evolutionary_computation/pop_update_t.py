# === update_population + should_stop sanity ===
import numpy as np
from update_population import update_population, pickup_best, should_stop

np.random.seed(0)

def mk_pop(fitness_list, L=6):
    N = len(fitness_list)
    tours = np.tile(np.array([0, 1, 2, 3, 4, 0]), (N, 1))  # dummy tours
    lengths = np.zeros(N, dtype=float)
    fitnesses = np.array(fitness_list, dtype=float)
    return (tours, fitnesses, lengths)

# Old population: fitness 1..5 (5 is the elite)
old = mk_pop([1, 2, 3, 4, 5])

# Offspring scenarios
offs_better_sameN = mk_pop([10, 9, 8, 7, 6])      # all children better than old
offs_mixed_bigger = mk_pop([3, 30, 2, 20, 1, 40]) # bigger than N
offs_small = mk_pop([2, 2, 2])                    # smaller than N (for elitism fallback)

# --- 1) offspring mode: keep best N children (when children >= N)
new_t, new_f, new_l = update_population(old, offs_mixed_bigger, mode="offspring")
assert new_t.shape[0] == old[0].shape[0]
# expect top-5 from offs_mixed_bigger: 40,30,20,3,2 (order by fitness desc)
assert list(new_f) == [40, 30, 20, 3, 2]
print("[offspring] OK: kept top N children.")

# --- 2) elitism mode: elites must always survive, even if all kids are better
new_t, new_f, new_l = update_population(old, offs_better_sameN, n_eliter=2, mode="elitism")
assert new_t.shape[0] == old[0].shape[0]
# elites from old are fitness 5 and 4; they must be present
assert 5 in new_f and 4 in new_f, "Elites were lost!"
print("[elitism] OK: elites preserved even when children are better.")

# --- 3a) elitism with children exactly equal to needed slots: NO filler from old.
new_t, new_f, new_l = update_population(old, offs_small, n_eliter=2, mode="elitism")
assert new_t.shape[0] == old[0].shape[0]
# elites 5,4 must be present; children fill the remaining 3 slots (all 2s); no '3' expected.
assert 5 in new_f and 4 in new_f
assert np.sum(new_f == 2) == 3
assert 3 not in new_f, "Non-elite old should NOT be used when children exactly fill the need."
print("[elitism] OK: exact-need case uses children only, no old filler.")

# --- 3b) elitism with children FEWER than needed: filler from best non-elite old expected.
offs_tiny = mk_pop([2, 2])  # need=3, children=2 -> filler=1 from old non-elites (best is 3)
new_t, new_f, new_l = update_population(old, offs_tiny, n_eliter=2, mode="elitism")
assert new_t.shape[0] == old[0].shape[0]
assert 5 in new_f and 4 in new_f
assert np.sum(new_f == 2) == 2
assert 3 in new_f, "Best non-elite old (3) should fill the remaining slot."
print("[elitism] OK: fewer-than-need case filled from best non-elite old.")


# --- 4) bestN mode: take top N from union
union = [1,2,3,4,5] + [10,9,8,7,6]
new_t, new_f, new_l = update_population(old, offs_better_sameN, mode="bestN")
assert list(new_f) == sorted(union, reverse=True)[:5]
print("[bestN] OK: selected global top N.")

# --- 5) pickup_best sanity
bt_t, bt_f, bt_l = pickup_best(old, 3)
assert list(bt_f) == [5,4,3]
print("[pickup_best] OK: top-k returned in desc order.")

# --- 6) should_stop sanity
assert should_stop(generation=10, evaluations=100, best_fitness=0.1,
                   max_generations=10, max_evaluations=200, target_fitness=None) is True
assert should_stop(generation=5, evaluations=100, best_fitness=0.9,
                   max_generations=10, max_evaluations=200, target_fitness=0.9) is True
assert should_stop(generation=5, evaluations=199, best_fitness=0.1,
                   max_generations=10, max_evaluations=200, target_fitness=None) is False
print("[should_stop] OK.")
