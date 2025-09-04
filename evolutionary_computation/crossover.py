import numpy as np


def ox_1child(tour1, tour2):
    """
    Order Crossover (OX) producing one child.
    - Assumes depot/city 0 fixed at start and end.
    - Cut points exclude endpoints [1 .. n-2].
    """
    n = len(tour1)
    assert n == len(tour2), "Parents must have same length."
    assert tour1[0] == 0 and tour1[-1] == 0 and tour2[0] == 0 and tour2[-1] == 0, \
        "Endpoints must be 0 in both parents."

    # Choose two cut points in the middle segment.
    cut_points = np.random.choice(range(1, n - 1), 2, replace=False)
    slice_st, slice_end = np.min(cut_points), np.max(cut_points)

    # Start with empty child, copy the slice from parent1.
    child = np.full(n, -1, dtype=int)
    child[slice_st:slice_end + 1] = tour1[slice_st:slice_end + 1]

    used = set(child[slice_st:slice_end + 1])  # cities already placed

    # Helpers to move within middle indices [1 .. n-2] circularly.
    def next_mid(idx):
        return 1 if idx >= (n - 2) else (idx + 1)

    # Phase 1: compute remaining cities in P2 wrap-order after the slice end.
    remaining = []
    rp = next_mid(slice_end)
    for _ in range(n - 2):  # scan all middle positions once
        city = tour2[rp]
        if city not in used:
            remaining.append(city)
        rp = next_mid(rp)

    # Phase 2: fill empty child slots in the same wrap-order after the slice end.
    wp = next_mid(slice_end)
    for city in remaining:
        while child[wp] != -1:
            wp = next_mid(wp)
        child[wp] = city
        wp = next_mid(wp)

    # Fix endpoints (depot).
    child[0], child[-1] = 0, 0
    return child


def ox2child(tour1, tour2):
    """Order Crossover (OX) producing two children."""
    return ox_1child(tour1, tour2), ox_1child(tour2, tour1)
