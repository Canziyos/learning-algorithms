import numpy as np

def pickup_best(population, k):
    tours, fitnesses, lengths = population
    # Get indices of top-k individuals by fitness (best first)
    best_idx = np.argsort(fitnesses)[-k:][::-1]
    return tours[best_idx], fitnesses[best_idx], lengths[best_idx]

def update_population(old_population, offspring, n_eliter=1, mode="elitism"):
    if mode == "offspring":
        # Replace entire population with offspring.
        return offspring

    elif mode == "elitism":
        # Elitism: keep top n_eliter from old + offspring.
        eli_tours, eli_fitnesses, eli_lengths = pickup_best(old_population, n_eliter)
        child_tours, child_fitnesses, child_lengths = offspring

        new_tours = np.vstack([eli_tours, child_tours])
        new_fitnesses = np.concatenate([eli_fitnesses, child_fitnesses])
        new_lengths = np.concatenate([eli_lengths, child_lengths])

        # Trim back to original size(if needed, of course).
        N = len(old_population[1])
        if len(new_fitnesses) > N:
            return pickup_best((new_tours, new_fitnesses, new_lengths), N)
        return new_tours, new_fitnesses, new_lengths

    elif mode == "bestN":
        # Combine old + offspring, then pick best N
        tours = np.vstack([old_population[0], offspring[0]])
        fitnesses = np.concatenate([old_population[1], offspring[1]])
        lengths = np.concatenate([old_population[2], offspring[2]])

        N = len(old_population[1])
        return pickup_best((tours, fitnesses, lengths), N)


def should_stop(generation, evaluations, best_fitness,
                    max_generations, max_evaluations, target_fitness=None):
    if target_fitness is not None and best_fitness >=target_fitness:
        return True
    
    if evaluations >= max_evaluations:
        return True
    
    if generation >= max_generations:
        return True
    
    return False
