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

OUT_NDJSON = "feasible_tours.ndjson"  # newline-delimited JSON (one record per line)

# ----------------------------
# GA runner (no local clean-up, no greedy seeding).
# Streams feasible tours to NDJSON if ndjson_fp + ndjson_config are provided.
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
    log_every=None,  # set to int to print occasional progress.
    ndjson_fp=None,  # file handle opened by caller; write one JSON line per feasible tour
    ndjson_config=None,  # dict with config + seed to include in each record
):
    if seed is not None:
        np.random.seed(seed)

    target_fitness = 1.0 / max_total_length
    max_evals = pop_size * (n_generations + 1)

    # Init.
    tours, fitnesses, lengths = init_population(pop_size, coords)
    best_idx = int(np.argmax(fitnesses))
    best_tour = tours[best_idx].copy()
    best_fitness = float(fitnesses[best_idx])
    evaluations = int(pop_size)
    gen = 0

    first_pass_gen = None
    first_pass_len = None

    # Feasible archive (dedup within *this run* by exact tour sequence)
    feasible_seen = set()  # keys are tuple(tour)
    feasible_count = 0

    if log_every:
        print(f"[gen {gen:3d}] best_len={lengths.min():.2f} avg_len={lengths.mean():.2f}.")

    # GA loop (do NOT stop on target_fitness; we only record it).
    while not should_stop(gen, evaluations, best_fitness, n_generations, max_evals, target_fitness=None):
        children_t, children_f, children_l = [], [], []

        while len(children_t) < pop_size:
            # Selection (probabilistic tournament).
            p_tours, _, _ = tournament_s(
                tours, fitnesses, lengths,
                n_parents=2, tournament_size=tournament_size,
                deterministic=False, p_win=tournament_win_prob
            )
            p1, p2 = p_tours[0], p_tours[1]

            # Light nudge against identical parents.
            tries = 0
            while np.array_equal(p1, p2) and tries < 3:
                p_redraw, _, _ = tournament_s(
                    tours, fitnesses, lengths,
                    n_parents=1, tournament_size=tournament_size,
                    deterministic=False, p_win=tournament_win_prob
                )
                p2 = p_redraw[0]
                tries += 1

            # Crossover (OX) with probability.
            if np.random.rand() < ox_prob:
                c1, c2 = ox2child(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation (inversion).
            c1 = mutate_inverse(c1, prob=mutation_prob)
            c2 = mutate_inverse(c2, prob=mutation_prob)

            # Evaluate kids.
            f1, l1 = tour_fitness(c1, coords)
            f2, l2 = tour_fitness(c2, coords)

            children_t.extend([c1, c2])
            children_f.extend([f1, f2])
            children_l.extend([l1, l2])
            evaluations += 2

        # Pack offspring and update with elitism.
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

        # Track best-of-gen and first time we pass the constraint.
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = float(fitnesses[gen_best_idx])
        gen_best_len = float(lengths[gen_best_idx])

        if gen_best_fit > best_fitness:
            best_fitness = gen_best_fit
            best_tour = tours[gen_best_idx].copy()

        if first_pass_gen is None and gen_best_fit >= target_fitness:
            first_pass_gen = gen
            first_pass_len = gen_best_len

        # Stream *all* feasible tours this generation (dedup within run)
        feas_mask = lengths <= (max_total_length + 1e-9)
        if np.any(feas_mask) and ndjson_fp is not None:
            feas_idx = np.where(feas_mask)[0]
            for idx in feas_idx:
                t = tours[idx]
                key = tuple(t.tolist())
                if key not in feasible_seen:
                    feasible_seen.add(key)
                    feasible_count += 1
                    rec = {
                        "config": ndjson_config or {},
                        "gen": int(gen),
                        "length": float(lengths[idx]),
                        "tour": t.tolist(),
                    }
                    ndjson_fp.write(json.dumps(rec) + "\n")
                    ndjson_fp.flush()

        if log_every and (gen % log_every == 0):
            print(f"[gen {gen:3d}] best_len={gen_best_len:.2f} avg_len={lengths.mean():.2f}.")

    best_len = 1.0 / best_fitness if best_fitness > 0 else float('inf')

    return {
        "best_len": best_len,
        "best_fitness": best_fitness,
        "best_tour": best_tour.tolist(),
        "first_pass_gen": first_pass_gen,
        "first_pass_len": first_pass_len,
        "generations": gen,
        "evaluations": evaluations,
        "pop_size": pop_size,
        "mutation_prob": mutation_prob,
        "ox_prob": ox_prob,
        "tournament_size": tournament_size,
        "tournament_win_prob": tournament_win_prob,
        "elite_size": elite_size,
        "n_generations": n_generations,
        "seed": seed,
        "feasible_count": feasible_count,
    }

# ----------------------------
# Grid search.
# ----------------------------
if __name__ == "__main__":
    path_to_berlin = "berlin52.tsp"
    data = parse_data(path_to_berlin)

    # grid.
    POP   = [200]
    PC    = [0.8, 0.9, 1.0]
    PM    = [0.05, 0.10]
    TK    = [2, 3, 4]
    PWIN  = [0.70, 0.75, 0.85]
    ELITE = [5, 10, 20]
    GENS  = [500]

    SEEDS = [0, 1, 2]

    configs = list(itertools.product(POP, PC, PM, TK, PWIN, ELITE, GENS))
    print(f"Total configs: {len(configs)}.")

    results = []
    t0 = time.time()

    # Truncate/overwrite NDJSON at start ( "a" to append across runs)
    with open(OUT_NDJSON, "w", encoding="utf-8") as ndj:
        for (pop, pc, pm, tk, pwin, elite, gens) in configs:
            bests = []
            first_pass_gens = []
            for s in SEEDS:
                cfg = {
                    "pop": pop, "pc": pc, "pm": pm, "tk": tk,
                    "pwin": pwin, "elite": elite, "gens": gens, "seed": s
                }
                out = run_ga(
                coords=data,
                pop_size=pop,
                mutation_prob=pm,
                ox_prob=pc,
                tournament_size=tk,
                tournament_win_prob=pwin,
                elite_size=elite,
                n_generations=gens,
                seed=s,
                log_every=None,
                ndjson_fp=ndj,              # stream feasible tours here
                ndjson_config=cfg,          # include config + seed in each record
            )
            bests.append(out["best_len"])
            if out["first_pass_gen"] is not None:
                first_pass_gens.append(out["first_pass_gen"])

            best_min = float(np.min(bests))
            best_mean = float(np.mean(bests))

            # Success stats (only successful seeds contribute to the mean/min)
            pass_rate = len(first_pass_gens) / len(SEEDS)
            pass_mean = float(np.mean(first_pass_gens)) if first_pass_gens else None
            pass_min  = int(np.min(first_pass_gens)) if first_pass_gens else None

            # Friendly display fields
            pass_mean_display = f"{np.mean(first_pass_gens):.1f}" if first_pass_gens else "—"
            pass_min_display  = str(pass_min) if pass_min is not None else "—"

            results.append({
                "pop": pop,
                "pc": pc,
                "pm": pm,
                "tk": tk,
                "pwin": pwin,
                "elite": elite,
                "gens": gens,
                "seeds": len(SEEDS),
                "best_min": best_min,
                "best_mean": best_mean,
                "first_pass_gen_mean": pass_mean,   # numeric for later sorting
                "first_pass_gen_min": pass_min,     # fastest crossing gen
                "pass_rate": pass_rate,             # fraction of seeds that passed
            })

            print(
                f"cfg pop={pop} pc={pc} pm={pm} tk={tk} pwin={pwin} elite={elite} "
                f"-> best_min={best_min:.2f} best_mean={best_mean:.2f} "
                f"pass_rate={pass_rate:.0%} first_pass_gen_mean={pass_mean_display} "
                f"(min={pass_min_display})."
            )

    # Show top 5 configs by best_min.
    results.sort(key=lambda r: r["best_min"])
    print("\nTop 5 configs by best_min.")
    for r in results[:5]:
        print(r)

    t1 = time.time()
    print(f"\nGrid search finished in {t1 - t0:.1f}s.")
    print(f"Feasible tours were streamed to {OUT_NDJSON} (one JSON object per line).")
