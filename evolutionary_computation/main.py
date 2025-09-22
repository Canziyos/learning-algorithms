# main.py
import numpy as np
from extract_data import parse_data
from utils import init_population, tour_fitness
from selection import tournament_s
from crossover import ox2child
from mutation import mutate_inverse
from update_pop import update_population, should_stop
import csv
import matplotlib.pyplot as plt

# Parameters.
path_to_berlin  = "berlin52.tsp"
max_total_length = 8000.0
# target_fitness = 1.0 / max_total_length

def run_ga_once(coords,
                pop_size, PC, PM,
                tournament_size, tournament_win_prob,
                elite_size, n_generations, seed):
    """Run GA once with the given config; returns (best_len, best_tour, best_gen)."""
    np.random.seed(seed)
    max_evals = pop_size * n_generations

    #Init.
    tours, fitnesses, lengths = init_population(pop_size, coords)  # fitness = 1/length.
    best_idx = int(np.argmax(fitnesses))
    best_fitness = float(fitnesses[best_idx])
    evaluations = int(pop_size)
    gen = 0
    history = []  # (generation, best_len, avg_len).
    history.append((gen, float(lengths.min()), float(lengths.mean())))
    # Track best feasible found in this run (≤ threshold).
    best_feas_len = float("inf")
    best_feas_tour = None
    best_feas_gen = None

    print(f"[gen {gen:3d}] best_len={lengths.min():.2f} avg_len={lengths.mean():.2f}.")

    # Check initial population for feasible solutions.
    feas_mask = lengths <= (max_total_length + 1e-9)
    if np.any(feas_mask):
        idx = int(np.argmin(lengths[feas_mask]))
        feas_idx = np.where(feas_mask)[0][idx]
        L0 = float(lengths[feas_idx])
        if L0 < best_feas_len:
            best_feas_len = L0
            best_feas_tour = tours[feas_idx].copy()
            best_feas_gen = gen

    # GA loop.
    while not should_stop(gen, evaluations, best_fitness, n_generations, max_evals, target_fitness=None):
        children_t, children_f, children_l = [], [], []

        # Produce exactly POP_SIZE children.
        while len(children_t) < pop_size:
            # Selection.
            p_tours, _, _ = tournament_s(
                tours, fitnesses, lengths,
                n_parents=2, tournament_size=tournament_size,
                deterministic=False, p_win=tournament_win_prob
            )
            p1, p2 = p_tours[0], p_tours[1]

            # Push uniquness of selected parents.
            tries = 0
            while np.array_equal(p1, p2) and tries < 3:
                redraw, _, _ = tournament_s(
                    tours, fitnesses, lengths,
                    n_parents=1, tournament_size=tournament_size,
                    deterministic=False, p_win=tournament_win_prob
                )
                p2 = redraw[0]
                tries += 1

            # Crossover.
            if np.random.rand() < PC:
                c1, c2 = ox2child(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation (inversion).
            c1 = mutate_inverse(c1, prob=PM)
            c2 = mutate_inverse(c2, prob=PM)

            # Evaluate children.
            f1, l1 = tour_fitness(c1, coords)
            f2, l2 = tour_fitness(c2, coords)

            children_t.extend([c1, c2])
            children_f.extend([f1, f2])
            children_l.extend([l1, l2])
            evaluations += 2

        # Check offspring for feasible before they’re possibly discarded.
        children_l = np.array(children_l, dtype=float)
        if np.any(children_l <= (max_total_length + 1e-9)):
            feas_idx = np.where(children_l <= (max_total_length + 1e-9))[0]
            best_child = feas_idx[int(np.argmin(children_l[feas_idx]))]
            Lc = float(children_l[best_child])
            if Lc < best_feas_len:
                best_feas_len = Lc
                best_feas_tour = children_t[best_child].copy()
                best_feas_gen = gen + 1

        # Update population (elitism). Trim to exactly pop_size.
        offspring = (
            np.array(children_t[:pop_size], dtype=int),
            np.array(children_f[:pop_size], dtype=float),
            np.array(children_l[:pop_size], dtype=float),
        )
        tours, fitnesses, lengths = update_population(
            (tours, fitnesses, lengths), offspring,
            n_eliter=elite_size, mode="elitism"
        )
        gen += 1

        # Track best of gen for fitness.
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = float(fitnesses[gen_best_idx])
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit

        # Also check updated population for feasible.
        feas_mask = lengths <= (max_total_length + 1e-9)
        if np.any(feas_mask):
            idx = int(np.argmin(lengths[feas_mask]))
            feas_idx = np.where(feas_mask)[0][idx]
            Lp = float(lengths[feas_idx])
            if Lp < best_feas_len:
                best_feas_len = Lp
                best_feas_tour = tours[feas_idx].copy()
                best_feas_gen = gen
        history.append((gen, float(lengths.min()), float(lengths.mean())))
        # Log.
        if gen % 50 == 0:
            print(f"[gen {gen:3d}] best_len={lengths.min():.2f} avg_len={lengths.mean():.2f}.")
        # Save performance trace and plot.
        csv_name = f"progress_seed{seed}.csv"
        png_name = f"progress_seed{seed}.png"
        with open(csv_name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["generation", "best_len", "avg_len"])
            w.writerows(history)

    plt.figure()
    gens = [g for g, _, _ in history]
    best = [b for _, b, _ in history]
    avg  = [a for _, _, a in history]

    plt.plot(gens, best, label="Best length")
    plt.plot(gens, avg,  label="Average length")

    # Feasibility threshold line.
    plt.axhline(max_total_length, linestyle="--", linewidth=1,
                label=f"Feasibility ≤ {int(max_total_length)}")

    # Mark the best feasible point.
    if best_feas_tour is not None:
        plt.scatter([best_feas_gen], [best_feas_len], s=36, zorder=3,
                    label=f"Best: {best_feas_len:.2f} @ gen {best_feas_gen}")
        # Optional annotation so the value is readable at a glance.
        plt.annotate(f"{best_feas_len:.0f}",
                    xy=(best_feas_gen, best_feas_len),
                    xytext=(best_feas_gen + 15, best_feas_len + 250),
                    arrowprops=dict(arrowstyle="->", lw=1),
                    fontsize=9)

    plt.xlabel("Generation")
    plt.ylabel("Tour length")
    plt.title(f"GA on Berlin52 (seed={seed}, pop={pop_size}, pc={PC}, pm={PM}, elites={elite_size})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_name, dpi=200)
    plt.close()

    return best_feas_len if best_feas_tour is not None else None, \
           (best_feas_tour.tolist() if best_feas_tour is not None else None), \
           (int(best_feas_gen) if best_feas_tour is not None else None)

# ----------------------------
# three best configs (from grid search log).
# ----------------------------
if __name__ == "__main__":
    coords = parse_data(path_to_berlin)

    configs = [
        # first best (7544.37), found at gen 587.
        {"pop": 200, "PC": 0.80, "PM": 0.05, "tournament_size": 4, "tournament_win_prob": 0.75, "elite_size": 10, "n_generations": 1000, "seed": 2},
            ]

    for i, cfg in enumerate(configs, 1):
        print(f"\n=== RUN {i}/3 ===")
        print(f"config = {cfg}.")
        best_len, best_tour, best_gen = run_ga_once(
            coords,
            cfg["pop"], cfg["PC"], cfg["PM"],
            cfg["tournament_size"], cfg["tournament_win_prob"],
            cfg["elite_size"], cfg["n_generations"], cfg["seed"]
        )
        print("\n=== RESULT ===")
        print(f"config = {cfg}.")
        if best_tour is not None:
            print(f"best_len = {best_len:.2f}  (≤ {int(max_total_length)}? {best_len <= max_total_length}).")
            print(f"found_at_gen = {best_gen}.")
            print("best_tour =", best_tour)
        else:
            print("No feasible tour found (no tour ≤ 8000).")
