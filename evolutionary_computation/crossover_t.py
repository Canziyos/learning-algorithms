import numpy as np
from crossover import ox_1child

def infer_slice_bounds(child, p1):
    # find the longest interior contiguous segment where child == p1
    n = len(child)
    best_len = 0
    best = (None, None)
    cur_len = 0
    cur_start = None
    for i in range(1, n-1):
        if child[i] == p1[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best = (cur_start, i)
        else:
            cur_len = 0
    return best  # (slice_st, slice_end)

def wrap_next(idx, n):
    return 1 if idx >= n-2 else idx+1

def expected_fill_order(p2, in_slice, start_after, n):
    # iterate parent2 middle indices in wrap order, skipping slice cities
    order = []
    rp = wrap_next(start_after, n)
    for _ in range(n-2):
        c = p2[rp]
        if c not in in_slice:
            order.append(c)
        rp = wrap_next(rp, n)
    return order

def child_fill_sequence(child, in_slice, start_after, n):
    seq = []
    wp = wrap_next(start_after, n)
    for _ in range(n-2):
        if child[wp] not in in_slice:  # positions outside the copied slice
            seq.append(child[wp])
        wp = wrap_next(wp, n)
    return seq

np.random.seed(7)

# small deterministic parents with depot 0
p1 = np.array([0, 1, 2, 3, 4, 5, 6, 0])
p2 = np.array([0, 6, 5, 4, 3, 2, 1, 0])

passes = 0
trials = 50
for _ in range(trials):
    child = ox_1child(p1, p2)
    n = len(child)

    # depot constraint.
    ok_depot = (child[0] == 0 and child[-1] == 0)

    # permutation validity (interior)
    interior = child[1:-1]
    ok_perm = (sorted(interior.tolist()) == list(range(1, n-1)))

    # infer copied slice and verify it matches p1
    st, en = infer_slice_bounds(child, p1)
    ok_slice = st is not None and np.all(child[st:en+1] == p1[st:en+1])

    # verify OX wrap-order from p2 for the remaining positions.
    in_slice = set(child[st:en+1])
    exp_order = expected_fill_order(p2, in_slice, en, n)
    got_order = child_fill_sequence(child, in_slice, en, n)
    ok_order = (exp_order == got_order)

    if ok_depot and ok_perm and ok_slice and ok_order:
        passes += 1

print(f"[OX] passed {passes}/{trials} trials")
