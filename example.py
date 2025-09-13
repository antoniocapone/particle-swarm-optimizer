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
