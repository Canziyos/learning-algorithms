import numpy as np
from crossover import ox2child
def infer_slice_bounds(child, p1):
    """Find the longest interior contiguous segment where child == p1."""
    n = len(child)
    best_len = 0
    best = (None, None)
    cur_len = 0
    cur_start = None
    for i in range(1, n - 1):  # skip endpoints.
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


def check_tour_valid(tour, n_cities):
    """
    Check if a tour is valid.
    - Must start and end with 0.
    - Must contain all cities [0..n_cities-1].
    - No duplicates except the endpoint 0.
    Returns True if valid, False otherwise.
    """
    if tour[0] != 0 or tour[-1] != 0:
        return False
    if len(tour) != n_cities + 1:
        return False
    if set(tour) != set(range(n_cities)).union({0}):
        return False
    if len(tour[1:-1]) != len(set(tour[1:-1])):
        return False
    return True

p1 = np.array([0, 1, 2, 3, 4, 5, 0])
p2 = np.array([0, 4, 3, 5, 1, 2, 0])

c1, c2 = ox2child(p1, p2)

print("Parent1:", p1)
print("Parent2:", p2)
print("Child1 :", c1)
print("Child2 :", c2)

# Infer slice copied from Parent1.
slice_bounds = infer_slice_bounds(c1, p1)
print("Child1 slice copied from P1:", slice_bounds)

slice_bounds = infer_slice_bounds(c2, p2)
print("Child2 slice copied from P2:", slice_bounds)
c1, c2 = ox2child(p1, p2)

print("Child1 valid:", check_tour_valid(c1, 6))
print("Child2 valid:", check_tour_valid(c2, 6))
