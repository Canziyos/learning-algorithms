
### Evolutionary Computation (EC)

* EC simulates **natural selection** in computer programs to improve performance automatically.
* Like nature, EC is a **stochastic search** in problem space.
* EC conducts **randomized, parallel beam search** (multiple candidates at once).
* Became more popular with modern computing power.

---

### EC: Population-Based vs Trajectory-Based Search

* **Trajectory-based search**: sequential, relies on local information (risk of local minima).
* **Population-based search**: explores globally using multiple candidates → stronger global ability.

**Examples of Evolutionary Algorithms:**

* Genetic Algorithms (GA).
* Genetic Programming (GP).
* Differential Evolution (DE).

---

### Population in a GA

* Population = many encoded solutions (chromosomes).
* Needs **coding/decoding** between string and real problem.
* Each chromosome has a **fitness value**.

---

### Flow Chart of GA

1. Create initial random population.
2. Evaluate fitness of each individual.
3. Termination criteria satisfied?

   * Yes → stop, return best one.
   * No → continue.
4. Select parents according to fitness (mating pool).
5. Recombine parents → offspring (crossover).
6. Mutate offspring.
7. Update population with new offspring.
8. Repeat.

---

### Key Issues in Designing GA

* How to **represent a solution** (chromosome coding).
* How to **select individuals** for mating.
* How to **create offspring** (crossover, mutation).
* How to **update population**.
* When to **terminate GA**.

---

### Coding Schemes

* Binary strings.
* Real-valued strings.
* (Later: Order representation, for permutation problems like TSP).

---

### Selection in GA

* **Darwinian principle**: fitter individuals more likely to reproduce.

Methods:

1. **Fitness-Proportionate (Roulette Wheel)**

   * Probability of selection ∝ fitness.

2. **Tournament Selection (TS)**

   * Pick k random individuals, select the best.

3. **Linear Order (LS)**

   * Rank by fitness, assign selection probabilities by order.

---

### Crossover

* Performed with probability **pc ∈ \[0.6, 1.0]**.
* Crossover site chosen randomly.

Types:

1. **One-Point Crossover**.

2. **Three-Point Crossover**.

3. **Arithmetic Crossover** (for real-coded):

   * Parents X, Y.
   * Offspring:

     * X′ = α₁X + (1−α₁)Y
     * Y′ = α₂X + (1−α₂)Y
   * α₁, α₂ ∈ \[0, 1] random.

4. **Crossover with Order Representation** (for permutations).

---

### Mutation

* Maintains **genetic diversity**.
* **Binary mutation**: flip a bit with probability **pm ∈ \[0.001, 0.1]**.
* **Order mutation**: pick two positions, swap their elements.
* **Real-coded mutation**: add random disturbance **uᵢ \~ N(0, δ)** to each gene.

---

### Updating Population

* Offspring replace entire population.
* **Elitism**: best individual(s) from old population carried forward.
* Or: select best N individuals from old + offspring.

---

### Stop Criteria

* Optimum or “good enough” solution found.
* Max number of fitness evaluations reached.
* Max number of generations reached.

---

### Genetic Programming (GP)

* Solutions = **programs** (trees).
* **Tree nodes** = functions (+, −, ×, ÷, sin, exp).
* **Leaves** = variables (X₁, X₂, …) or constants.

Operators:

* **Crossover**: exchange random subtrees.
* **Mutation**: change a function or terminal randomly.

---

### Differential Evolution (DE)

* Powerful algorithm for **real-valued optimization**.
* Population = vectors of real numbers (candidate solutions).

**Steps:**

1. For each target vector Xᵢ: select three distinct others (Xr₁, Xr₂, Xr₃).
2. Donor vector: V = Xr₁ + F·(Xr₂ − Xr₃), with F ∈ \[0, 2].
3. Crossover: combine Xᵢ and V to create trial vector Tᵢ.
4. Evaluate fitness of Tᵢ. If better than Xᵢ, replace it.

**Parameters:**

* CR = crossover rate \[0,1].
* F = mutation factor \[0,2].

---

### References / Extra Reading

* Russell & Norvig (2010): *Artificial Intelligence: A Modern Approach*.
* Tom Mitchell (ML textbook, Ch. 9).
* Wikipedia: Genetic Algorithm.
* Storn & Price (1997): Differential Evolution.
* Miguel Leon: *Improving Differential Evolution with Adaptive and Local Search Methods* (DiVA portal).



## **Assignment 2 – The Travelling Salesman Problem (TSP)**

### Problem Statement

* The Travelling Salesman Problem (TSP) asks:
  *Given a list of cities and the distances between them, what is the shortest route that visits all cities once and returns to the starting city?*

* In this assignment:

  * Dataset: **Berlin52.tsp** (locations with coordinates x, y).
  * Example:

    ```
    Location ID   x     y
    1             565   575
    2              25   185
    3             345   750
    …             …     …
    ```

* Distance metric: **Euclidean Distance**

  $$
  D((x₁,y₁),(x₂,y₂)) = \sqrt{(x₁ - x₂)^2 + (y₁ - y₂)^2}
  $$

---

### Requirements

* Write a program using an **optimization algorithm** (e.g., Genetic Algorithm).
* Constraints:

  * All locations must be visited **once and only once**.
  * The **starting and ending location = location 1**.
  * Total distance of the route must be **≤ 8000**.

---

### Report Structure

The report must include:

1. **Explanation of the employed algorithm**

   * Describe GA (or other chosen algorithm) and how it works for this problem.

2. **Representation of individual solutions**

   * How each route (tour) is encoded in your algorithm.

3. **Fitness function**

   * State the formula used to evaluate solutions (e.g., based on total distance).

4. **Parameters of the algorithm**

   * Population size, crossover rate, mutation rate, number of generations, etc.

5. **Performance evolution**

   * Show how the population improves over generations, with a figure/plot.

6. **Best result obtained**

   * Show the best route found:

     * Order of locations visited.
     * Total distance of this route.

---
