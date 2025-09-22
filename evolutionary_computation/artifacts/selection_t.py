import numpy as np
from selection import touranment_s

np.random.seed(42)

# 5 tours, with arbitrary "fitness" values
tours = np.array([
    [0, 1, 2, 3, 0],
    [0, 2, 3, 1, 0],
    [0, 3, 1, 2, 0],
    [0, 1, 3, 2, 0],
    [0, 2, 1, 3, 0],
])
fitnesses = np.array([0.1, 0.5, 0.3, 0.05, 0.2])  # higher is better.
lengths = 1.0 / fitnesses  # inverse for test.

# Deterministic tournament: k=3
selected_tours, selected_fitness, selected_lengths = touranment_s(
    tours, fitnesses, lengths,
    n_parents=5,
    touranment_size=3,
    deterministic=True
)

print("Deterministic TS:")
print("Tours:\n", selected_tours)
print("Fitnesses:", selected_fitness)
print("Lengths:", selected_lengths)

# Probabilistic tournament: k=3, p_win=0.7
selected_tours_prob, selected_fitness_prob, selected_lengths_prob = touranment_s(
    tours, fitnesses, lengths,
    n_parents=5,
    touranment_size=3,
    deterministic=False,
    p_win=0.7
)

print("\nProbabilistic TS:")
print("Tours:\n", selected_tours_prob)
print("Fitnesses:", selected_fitness_prob)
print("Lengths:", selected_lengths_prob)


# # dummy population (controled fitness).
# N = 10
# L = 6
# tours = np.tile(np.array([0, 1, 2, 3, 4, 0]), (N, 1))  # tours with correct shape.
# lengths = np.zeros(N)  # irrelevant for selection tests.
# fitnesses = np.linspace(1.0, 10.0, N)  # 1..10, so index 9 is the best.

# pop_mean_fit = fitnesses.mean()
# pop_best_fit = fitnesses.max()

# # LRS test: many draws, mean selected fitness should be > population mean.
# lrs_draws = 2000
# lrs_selected_fits = []
# for _ in range(lrs_draws):
#     _, f, _ = lrs(tours, fitnesses, lengths, sel_pres=1.7)
#     lrs_selected_fits.append(f)
# lrs_selected_fits = np.array(lrs_selected_fits)

# print(f"[LRS] mean_selected_fit = {lrs_selected_fits.mean():.3f} | pop_mean = {pop_mean_fit:.3f} | pop_best = {pop_best_fit:.3f}")
# print(f"[LRS] fraction of best picked ≈ {np.mean(lrs_selected_fits == pop_best_fit):.3f}")

# # RWS: many draws, empirical prob(best) approx fitness(best) / sum(fitness)
# rws_draws = 2000
# rws_selected_fits = []
# for _ in range(rws_draws):
#     _, f, _ = rws(tours, fitnesses, lengths)
#     rws_selected_fits.append(f)
# rws_selected_fits = np.array(rws_selected_fits)

# theoretical_best_p = pop_best_fit / fitnesses.sum()
# empirical_best_p = np.mean(rws_selected_fits == pop_best_fit)
# print(f"[RWS] empirical_best_p = {empirical_best_p:.3f} | theoretical_best_p = {theoretical_best_p:.3f}")
# print(f"[RWS] mean_selected_fit = {rws_selected_fits.mean():.3f} | pop_mean = {pop_mean_fit:.3f}")
