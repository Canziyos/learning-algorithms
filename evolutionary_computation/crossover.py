import numpy as np


def ox_1child(tour1, tour2):
    cut_points = np.random.choice(range(1, len(tour1)-1), 2, replace=False)
    slice_st, slice_end = np.min(cut_points), np.max(cut_points)
    child = np.full(tour1.shape[0], -1, dtype=int)

    # Copy slice from parent1
    child[slice_st:slice_end+1] = tour1[slice_st:slice_end+1]

    # Fill the remaining slots from parent2, skipping duplicates
    for city in tour2[1:-1]:  # exclude start/end
        if city not in child:
            for j in range(1, len(child)-1):  # only middle positions--
                if child[j] == -1:
                    child[j] = city
                    break

    # Ensure start/end = 0
    child[0], child[-1] = 0, 0
    return child

def ox2child(tour1, tour2):
    child1 = ox_1child(tour1, tour2)
    child2 = ox_1child(tour2, tour1)

    return child1, child2


