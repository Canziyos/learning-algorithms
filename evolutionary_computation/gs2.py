import itertools
import time
import json
import numpy as np

from extract_data import parse_data
from utils import init_population, tour_fitness
from selection import tournament_s
from crossover import ox2child
from mutation import mutate_inverse
from update_pop import update_population, should_stop

# to record the configs that produced a feasible tour (≤ 8000).
OUT_JSON = "survivors.json"   


# Runner: returns the best feasibale (≤ max_total_length) found in the run.
# ----------------------------
def run_ga(
    coords,
    pop_size=200,
    mutation_prob=0.10,
    ox_prob=0.90,
    tournament_size=3,
    tournament_win_prob=0.75,
    elite_size=10,
    n_generations=500,
    max_total_length=8000.0,
    seed=None,
    log_every=None,
):
    if seed is not None:
        np.random.seed(seed)
    # Uncomment when introducing early stop, should stop when fitness is reached.
    # target_fitness = 1.0 / max_total_length
    max_evals = pop_size * (n_generations + 1) # including init batch.

    # Init
    tours, fitnesses, lengths = init_population(pop_size, coords)
    best_idx = int(np.argmax(fitnesses))
    best_fitness = float(fitnesses[best_idx])
    evaluations = int(pop_size)
    gen = 0

    # Track best feasible found in this run (≤ threshold).
    best_feas_len = float("inf")
    best_feas_tour = None
    best_feas_gen = None

    if log_every:
        print(f"[gen {gen:3d}] best_len={lengths.min():.2f} avg_len={lengths.mean():.2f}")

    # Check initial population for feasible solutions.
    feas_mask = lengths <= (max_total_length + 1e-9)

    if np.any(feas_mask):
        # find the idx of the shortest tour among the feasible ones.
        idx = int(np.argmin(lengths[feas_mask]))

        # Map that idx back to the original population idx.
        feas_idx = np.where(feas_mask)[0][idx]

        # grabs that tour’s length and stores it in the following:
        _len = float(lengths[feas_idx])
        if _len < best_feas_len:
            best_feas_len = _len
            best_feas_tour = tours[feas_idx].copy()
            best_feas_gen = gen

    # GA loop (no stop on target; collect the best feasible).
    while not should_stop(gen, evaluations, best_fitness, n_generations, max_evals, target_fitness=None):
        children_t, children_f, children_l = [], [], []

        while _len(children_t) < pop_size:
            # Selection
            p_tours, _, _ = tournament_s(tours, fitnesses, lengths,
                n_parents=2, tournament_size=tournament_size,
                deterministic=False, p_win=tournament_win_prob
            )
            p1, p2 = p_tours[0], p_tours[1]

            # This is to push uniquness of selected parents.
            # If, by bad luck, after 3 attempts they’re still identical,
            # => just accept it and move on.
            tries = 0
            while np.array_equal(p1, p2) and tries < 3:
                p2, _, _ = tournament_s(
                    tours, fitnesses, lengths,
                    n_parents=1, tournament_size=tournament_size,
                    deterministic=False, p_win=tournament_win_prob
                )
                p2 = p2[0]
                tries += 1

            # Crossover
            if np.random.rand() < ox_prob:
                c1, c2 = ox2child(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation (inversion)
            c1 = mutate_inverse(c1, prob=mutation_prob)
            c2 = mutate_inverse(c2, prob=mutation_prob)

            # Evaluate children
            f1, l1 = tour_fitness(c1, coords)
            f2, l2 = tour_fitness(c2, coords)

            children_t.extend([c1, c2])
            children_f.extend([f1, f2])
            children_l.extend([l1, l2])
            evaluations += 2

        # Check offspring for feasible before they're possibly discarded.
        children_l = np.array(children_l, dtype=float)
        if np.any(children_l <= (max_total_length + 1e-9)):
            feas_idx = np.where(children_l <= (max_total_length + 1e-9))[0]
            # get the best feasible child.
            best_child = feas_idx[int(np.argmin(children_l[feas_idx]))]
            len_ch = float(children_l[best_child])
            if len_ch < best_feas_len:
                best_feas_len = len_ch
                best_feas_tour = children_t[best_child].copy()
                best_feas_gen = gen + 1  # next generation (offspring produced in this gen)

        # Update population, but before doing that, we trim to exactly pop_size. (might not be divisible by 2)
        # During the loop, we may have collected a little more than pop_size children.
        # (depending on how crossover pairs fit).
        # finally; pack them into a pop tuple (tours, fitnesses, lengths),
        # so update_population can handle them.
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

        # Track best of the gen for fitness. (Here this (best fitness <=> feasible) is false.
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = float(fitnesses[gen_best_idx])
        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit

        # Also check updated population for feasible
        feas_mask = lengths <= (max_total_length + 1e-9)
        if np.any(feas_mask):
            idx = int(np.argmin(lengths[feas_mask]))
            feas_idx = np.where(feas_mask)[0][idx]
            _len = float(lengths[feas_idx])
            if _len < best_feas_len:
                best_feas_len = _len
                best_feas_tour = tours[feas_idx].copy()
                best_feas_gen = gen

        if log_every and (gen % log_every == 0):
            print(f"[gen {gen:3d}] best_len={lengths.min():.2f} avg_len={lengths.mean():.2f}")

    return {
        "best_feasible_len": (best_feas_len if best_feas_tour is not None else None),
        "best_feasible_tour": (best_feas_tour.tolist() if best_feas_tour is not None else None),
        "best_feasible_gen": (int(best_feas_gen) if best_feas_gen is not None else None),
    }


# Grid search, gs, saves only survivors to OUT_JSON.
# ----------------------------
if __name__ == "__main__":
    path_to_berlin = "berlin52.tsp"
    data = parse_data(path_to_berlin)

    # Focused grid around our winners
    POP = [200]
    PC = [0.8, 0.9]
    PM = [0.05, 0.06, 0.09, 0.10]
    TK = [4]
    PWIN = [0.75, 0.85]
    ELITE = [10, 15, 20]
    GENS = [800, 1000]
    SEEDS = [0, 1, 2]

    configs = list(itertools.product(POP, PC, PM, TK, PWIN, ELITE, GENS))
    print(f"Total configs: {len(configs)}.")

    survivors = []       # one record per CONFIG (best across seeds), only if any seed survived.
    global_best = None   # best across all configs+seeds.

    t0 = time.time()
    for (pop, pc, pm, tk, pwin, elite, gens) in configs:
        config_best = None  # best survivor across seeds for this config.

        for s in SEEDS:
            out = run_ga(coords=data, pop_size=pop, mutation_prob=pm,
                ox_prob=pc, tournament_size=tk, tournament_win_prob=pwin,
                elite_size=elite, n_generations=gens, seed=s, log_every=None,
            )

            if out["best_feasible_len"] is None:
                continue  # this seed didn’t survive.
            # into json.
            rec = {
                "config": {"pop": pop, "pc": pc, "pm": pm, "tk": tk,
                           "pwin": pwin, "elite": elite, "gens": gens, "seed": s},
                "gen": out["best_feasible_gen"],
                "length": out["best_feasible_len"],
                "tour": out["best_feasible_tour"],
            }

            # per-config best (across seeds).
            if (config_best is None) or (rec["length"] < config_best["length"]):
                config_best = rec

            # global best.
            if (global_best is None) or (rec["length"] < global_best["length"]):
                global_best = rec

        # store per-config record only if some seed survived.
        if config_best is not None:
            survivors.append(config_best)
            print(f"config survived: {config_best['config']} len={config_best['length']:.2f} gen={config_best['gen']}")

    # Save only survivors (one per config) to a JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(survivors, f, indent=2)
    print(f"\nSaved {len(survivors)} survivors (one per config) to {OUT_JSON}")

    # Show the global best, (overal view).
    if global_best:
        print("\nGLOBAL BEST")
        print(json.dumps(global_best, indent=2))
    else:
        print("\nNo survivors (no tour ≤ 8000 found).")

    t1 = time.time()
    print(f"\nGrid search finished in {t1 - t0:.1f}s.")
