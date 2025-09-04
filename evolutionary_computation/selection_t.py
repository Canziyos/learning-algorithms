import numpy as np
from selection import lrs, rws, ts

np.random.seed(42)

# Build a tiny dummy population where we control fitness directly.
N = 10
L = 6
tours = np.tile(np.array([0, 1, 2, 3, 4, 0]), (N, 1))  # dummy tours with correct shape.
lengths = np.zeros(N)  # irrelevant for selection tests.
fitnesses = np.linspace(1.0, 10.0, N)  # 1..10, so index 9 is the best.

pop_mean_fit = fitnesses.mean()
pop_best_fit = fitnesses.max()

# --- LRS test: many draws, mean selected fitness should be > population mean.
lrs_draws = 2000
lrs_selected_fits = []
for _ in range(lrs_draws):
    _, f, _ = lrs(tours, fitnesses, lengths, sel_pres=1.7)
    lrs_selected_fits.append(f)
lrs_selected_fits = np.array(lrs_selected_fits)

print(f"[LRS] mean_selected_fit = {lrs_selected_fits.mean():.3f} | pop_mean = {pop_mean_fit:.3f} | pop_best = {pop_best_fit:.3f}")
print(f"[LRS] fraction of best picked ≈ {np.mean(lrs_selected_fits == pop_best_fit):.3f}")

# --- RWS test: many draws, empirical prob(best) ≈ fitness(best) / sum(fitness)
rws_draws = 2000
rws_selected_fits = []
for _ in range(rws_draws):
    _, f, _ = rws(tours, fitnesses, lengths)
    rws_selected_fits.append(f)
rws_selected_fits = np.array(rws_selected_fits)

theoretical_best_p = pop_best_fit / fitnesses.sum()
empirical_best_p = np.mean(rws_selected_fits == pop_best_fit)
print(f"[RWS] empirical_best_p = {empirical_best_p:.3f} | theoretical_best_p = {theoretical_best_p:.3f}")
print(f"[RWS] mean_selected_fit = {rws_selected_fits.mean():.3f} | pop_mean = {pop_mean_fit:.3f}")
