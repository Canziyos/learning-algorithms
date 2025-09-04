import numpy as np

def mutate_swap(tour, n_swaps=1, prob=0.1):
    # With probability prob, perform swap mutation
    if np.random.rand() < prob:
        copied_tour = np.copy(tour)
        for _ in range(n_swaps):
            # Pick two distinct interior positions (exclude start=0, end=last)
            two_points = np.random.choice(range(1, len(tour)-1), 2, replace=False)
            # Swap the two selected cities
            copied_tour[[two_points[0], two_points[1]]] = copied_tour[[two_points[1], two_points[0]]]
        return copied_tour
    else:
        # No mutation, return original
        return tour


def mutate_insertion(tour, prob=0.1):
    if np.random.rand() < prob:
        copied = np.copy(tour)
        rand_pos = np.random.choice(range(1, len(tour)-1), 2, replace=False)
        i, j = rand_pos
        
        city = copied[i]
        tmp = np.delete(copied, i)
        if j>=i:
            j -= 1   # adjustment for shifted array.
        mutated = np.insert(tmp, j, city)

        return mutated
    else:
        return tour
    
def mutate_inversion(tour, prob=0.1):
    if np.random.rand() <= prob:
        copied_t = np.copy(tour)
        i, j = np.random.choice(range(1, len(tour)-1), 2, replace=False)
        if i > j:
            i, j = j, i  # ensure i < j
        # Reverse the slice between i and j.
        copied_t[i:j+1] = copied_t[i:j+1][::-1]
        return copied_t
    else:
        return tour


