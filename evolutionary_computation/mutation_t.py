# === Mutation sanity test (swap / insertion / inversion) ===
import numpy as np
from mutation import mutate_swap, mutate_insertion, mutate_inversion

np.random.seed(123)

def valid_tour(t):
    return t[0] == 0 and t[-1] == 0 and sorted(t[1:-1].tolist()) == list(range(1, len(t)-1))

# Base tour (small, deterministic)
base = np.array([0, 1, 2, 3, 4, 5, 6, 0])

# 1) No-mutation path returns the same object (copy-on-mutate semantics)
t1 = base.copy()
t2 = base.copy()
t3 = base.copy()
assert mutate_swap(t1, prob=0.0) is t1, "swap: should return original object when prob=0"
assert mutate_insertion(t2, prob=0.0) is t2, "insertion: should return original object when prob=0"
assert mutate_inversion(t3, prob=0.0) is t3, "inversion: should return original object when prob=0"

# 2) prob=1 path keeps a valid tour; across trials, at least one actual change occurs
def run_trials(mutator, name, trials=50):
    changed = 0
    for _ in range(trials):
        m = mutator(base, prob=1.0)
        assert valid_tour(m), f"{name}: produced invalid tour"
        if not np.array_equal(m, base):
            changed += 1
    print(f"[{name}] valid={True} | changed_at_least_once={(changed > 0)} | changes={changed}/{trials}")

run_trials(mutate_swap, "swap")
run_trials(mutate_insertion, "insertion")
run_trials(mutate_inversion, "inversion")
