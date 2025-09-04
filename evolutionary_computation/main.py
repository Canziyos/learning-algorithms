from update_pop import should_stop

print("\n=== Testing should_stop ===")

# Case 1: stop because best fitness reached target
print("Case 1:", should_stop(
    generation=5, evaluations=100, best_fitness=0.9,
    max_generations=50, max_evaluations=1000, target_fitness=0.8
))  # Expect True

# Case 2: stop because evaluations exceeded
print("Case 2:", should_stop(
    generation=5, evaluations=1200, best_fitness=0.5,
    max_generations=50, max_evaluations=1000, target_fitness=0.8
))  # Expect True

# Case 3: stop because generations exceeded
print("Case 3:", should_stop(
    generation=60, evaluations=500, best_fitness=0.5,
    max_generations=50, max_evaluations=1000, target_fitness=0.8
))  # Expect True

# Case 4: none of the criteria met
print("Case 4:", should_stop(
    generation=10, evaluations=200, best_fitness=0.5,
    max_generations=50, max_evaluations=1000, target_fitness=0.8
))  # Expect False
