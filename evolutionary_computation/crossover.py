import numpy as np

def ox_1child(tour1, tour2):
    """
    Order Crossover (OX).
    One child from two parents.
    - City 0 fixed at start and end.
    - Cut points exclude endpoints [1 .. n-2].
    """
    n = len(tour1)
    assert n == len(tour2), "Parents must have same length."
    assert tour1[0] == 0 and tour1[-1] == 0 and tour2[0] == 0 and tour2[-1] == 0, \
        "Endpoints must be 0 in both parents."

    # Pick 2 random cut points in middle.
    slice_st, slice_end = sorted(np.random.choice(range(1, n - 1), 2, replace=False))

    # Copy slice from parent1 into child.
    child = np.full(n, -1, dtype=int)
    child[slice_st:slice_end + 1] = tour1[slice_st:slice_end + 1]

    # Mark used cities.
    used = set(child[slice_st:slice_end + 1])

    # Helper: move in middle [1 .. n-2], wrap-around.
    def next_mid(idx):
        return 1 if idx >= (n - 2) else (idx + 1)

    # Collect cities from parent2 after slice_end, skipping used.
    remaining = []
    rp = next_mid(slice_end)
    for _ in range(n - 2):
        city = tour2[rp]
        if city not in used:
            remaining.append(city)
        rp = next_mid(rp)

    # Fill empty slots in child in same wrap order.
    wp = next_mid(slice_end)
    for city in remaining:
        while child[wp] != -1:
            wp = next_mid(wp)
        child[wp] = city
        wp = next_mid(wp)

    # Fix endpoints.
    child[0], child[-1] = 0, 0
    return child

def ox2child(tour1, tour2):
    """OX producing two children."""
    return ox_1child(tour1, tour2), ox_1child(tour2, tour1)
