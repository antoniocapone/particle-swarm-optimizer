# Particle Swarm Optimization (PSO) – Python Implementation

This project provides a clean and extensible **Particle Swarm Optimization (PSO)** implementation in pure Python and NumPy.
It is designed for research, teaching, and experimentation with continuous, possibly multi-modal optimization problems.

---

## Overview

Particle Swarm Optimization is a population-based metaheuristic inspired by the collective motion of birds and fish.
A swarm of candidate solutions (particles) moves through the search space, combining:

* **Inertia** – keeps the particle moving in its current direction.
* **Cognitive component** – pulls the particle toward its own best position.
* **Social component** – pulls it toward the best global position found by the swarm.

This implementation supports arbitrary search-space dimension and user-defined objective (fitness) functions.

---

## Key Features

- Object-oriented, modular design (`Particle`, `ParticleSwarmOptimizer`).
- Fully configurable hyperparameters:
  - inertia weight `w`
  - cognitive and social coefficients `c1` and `c2`
  - maximum velocity as fraction of domain size.
- Optional linear scheduling of the inertia weight and early stopping based on convergence.
- Built-in handling of box constraints (variable-wise lower and upper bounds).
- Reproducible runs via random seed.

---

## Quick Example

```python
import numpy as np
from pso import ParticleSwarmOptimizer

# Objective: minimum at x = 10
def fitness(x: np.ndarray) -> float:
	return float((x[0] - 10.0)**2)

def main() -> None:
	bounds = np.array([[0.0, 20.0]])  # search domain
	optimizer = ParticleSwarmOptimizer(
		fitness_f=fitness,
		domain_dim=1,
		particle_num=30,
		bounds=bounds,
		seed=0
	)

	best_pos = optimizer.optimize(num_iterations=200, verbose=True)
	print("Best position:", best_pos, "fitness:", optimizer.gbest_val)

if __name__ == "__main__":
	main()
```
