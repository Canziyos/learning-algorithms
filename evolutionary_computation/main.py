import numpy as np
from extract_data import parse_data
from utils import init_population, tour_fitness
from selection import ts
from crossover import ox2child
from mutation import mutate_inversion, mutate_insertion, mutate_swap

from update_population import update_population, should_stop



# Parameters. 
PATH_TO_BERLIN  = "berlin52.tsp"
THRESHOLD = 8000.0            # used only for stopping via target_fitness
POP_SIZE  = 200
PC = 1.00              # always crossover (strong exploitation)..
PM = 0.40              # inversion mutation probability per child
TOURNAMENT_K = 5                 # deterministic tournament size.
ELITISM_K = 5
MAX_GENERATIONS = 500
MAX_EVALUATIONS = POP_SIZE * MAX_GENERATIONS
TARGET_FITNESS  = 1.0 / THRESHOLD   # fitness = 1/length

#np.random.seed(0)


# === Init ===
data = parse_data(PATH_TO_BERLIN)
tours, fitnesses, lengths = init_population(POP_SIZE, data)  # fitness = 1/length in utils
best_idx = int(np.argmax(fitnesses))
best_tour = tours[best_idx].copy()
best_fitness = float(fitnesses[best_idx])
evaluations = int(POP_SIZE)
gen = 0

print(f"[gen {gen:3d}] best_len={lengths.min():.2f} avg_len={lengths.mean():.2f}")


# === GA loop ===
while not should_stop(gen, evaluations, best_fitness,
                      MAX_GENERATIONS, MAX_EVALUATIONS, TARGET_FITNESS):
    children_t, children_f, children_l = [], [], []
    mutated = 0  # count across the whole generation.

    # Produce exactly POP_SIZE children..
    while len(children_t) < POP_SIZE:
        # Parent selection via deterministic Tournament Selection.
        p_tours, _, _ = ts(tours, fitnesses, lengths,
                           n_parents=2, n_candidates=TOURNAMENT_K, deterministic=True)
        p1, p2 = p_tours[0], p_tours[1]

        # avoid identical parents (light diversity nudge).
        if np.array_equal(p1, p2):
            p_redraw, _, _ = ts(tours, fitnesses, lengths,
                                n_parents=1, n_candidates=TOURNAMENT_K, deterministic=True)
            p2 = p_redraw[0]

        # Crossover (OX).
        if np.random.rand() < PC:
            c1, c2 = ox2child(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()

        # Invariants (can comment out later).
        #assert c1[0] == 0 and c1[-1] == 0
        #assert c2[0] == 0 and c2[-1] == 0

        # Mutation (inversion)
        pre1, pre2 = c1.copy(), c2.copy()
        c1 = mutate_insertion(c1, prob=PM)
        c2 = mutate_inversion(c2, prob=PM)
        if not np.array_equal(pre1, c1): mutated += 1
        if not np.array_equal(pre2, c2): mutated += 1

        # Evaluate children.
        f1, l1 = tour_fitness(c1, data)   # fitness = 1/length
        f2, l2 = tour_fitness(c2, data)

        children_t.extend([c1, c2])
        children_f.extend([f1, f2])
        children_l.extend([l1, l2])
        evaluations += 2

    # Pack offspring.
    offspring = (np.array(children_t[:POP_SIZE], dtype=int),
                 np.array(children_f[:POP_SIZE], dtype=float),
                 np.array(children_l[:POP_SIZE], dtype=float))

    # --- Update (elitism) ---
    tours, fitnesses, lengths = update_population(
        (tours, fitnesses, lengths), offspring,
        n_eliter=ELITISM_K, mode="elitism"
    )
    gen += 1

    # Track best-of-run
    gen_best_idx = int(np.argmax(fitnesses))
    gen_best_fit = float(fitnesses[gen_best_idx])
    gen_best_len = float(lengths[gen_best_idx])
    if gen_best_fit > best_fitness:
        best_fitness = gen_best_fit
        best_tour = tours[gen_best_idx].copy()

    # Light logging every 10 gens or when threshold reached.
    if gen % 10 == 0 or gen_best_fit >= TARGET_FITNESS:
        print(f"[gen {gen:3d}] best_len={gen_best_len:.2f} avg_len={lengths.mean():.2f} mutated={mutated}/{POP_SIZE}")

    if gen_best_fit >= TARGET_FITNESS:
        break


# === Result ===
print("\n=== RESULT ===")
print(f"generations = {gen}, evaluations = {evaluations}")
if best_fitness > 0:
    best_len = 1.0 / best_fitness
    print(f"best_len = {best_len:.2f}  (≤ {int(THRESHOLD)}? {best_len <= THRESHOLD})")
else:
    print("best_len = (no feasible tour yet; fitness stayed 0)")
print("best_tour =", best_tour.tolist())
