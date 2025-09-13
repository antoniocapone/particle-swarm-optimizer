from typing import Callable, Optional, Tuple, List
import numpy as np
from particle import Particle


class ParticleSwarmOptimizer:
	def __init__(
		self,
		fitness_f: Callable[[np.ndarray], float],
		domain_dim: int,
		particle_num: int,
		inertia_weight: Optional[float] = None,
		cognitive_coeff: Optional[float] = None,
		social_coeff: Optional[float] = None,
		bounds: Optional[np.ndarray] = None,      # shape (D, 2)
		v_max_frac: float = 0.3,
		seed: Optional[int] = None,
	) -> None:
		if fitness_f is None or domain_dim is None:
			raise RuntimeError("Fitness function or domain dimension cannot be null")

		if domain_dim <= 0:
			raise ValueError("Problem dimension must be greater than 0")

		if particle_num <= 0:
			raise ValueError("Particle number must be greater than 0")

		self.fitness_f = fitness_f
		self.D = int(domain_dim)
		self.N = int(particle_num)

		# TODO: choose better default values
		self.w  = float(inertia_weight) if inertia_weight is not None else np.random.uniform(0.4, 0.9)
		self.c1 = float(cognitive_coeff) if cognitive_coeff is not None else 2.0
		self.c2 = float(social_coeff) if social_coeff is not None    else 2.0

		if bounds is None:
			self.bounds = np.column_stack([np.zeros(self.D), np.ones(self.D)]).astype(float)
		else:
			self.bounds = np.asarray(bounds, dtype=float)
			if self.bounds.shape != (self.D, 2):
				raise ValueError("Bounds must have shape (domain_dim, 2)")

		# Random number generator
		self.rng = np.random.default_rng(seed)

		# Create particles
		self.particles: List[Particle] = []
		for _ in range(self.N):
			p = Particle(
				domain_dim=self.D,
				bounds=self.bounds,
				inertia_weight=self.w,
				cognitive_coeff=self.c1,
				social_coeff=self.c2,
				v_max_frac=v_max_frac,
				rng=self.rng,
				fitness_f=self.fitness_f,
			)
			self.particles.append(p)

		# Compute first gbest
		pbest_vals = np.array([p.get_pbest_val() for p in self.particles], dtype=float)
		best_idx = int(np.argmin(pbest_vals))
		self.gbest_pos: np.ndarray = self.particles[best_idx].get_pbest_pos().copy()
		self.gbest_val: float = float(self.particles[best_idx].get_pbest_val())

	def optimize(
		self,
		num_iterations: int,
		tol: Optional[float] = None,
		patience: Optional[int] = None,
		w_schedule: Optional[Tuple[float, float]] = None,  # (w_start, w_end) lineare
		verbose: bool = False,
	) -> np.ndarray:
		if num_iterations is None or num_iterations <= 0:
			raise ValueError("Iterations number must be greater than 0")

		no_improve = 0
		best_val_prev = self.gbest_val

		for it in range(num_iterations):
			if w_schedule is not None:
				w_start, w_end = w_schedule
				t = it / max(1, num_iterations - 1)
				self.w = (1 - t) * w_start + t * w_end
				# Update w in particles
				for p in self.particles:
					p.w = self.w

			improved = False

			for p in self.particles:
				pbest_pos, pbest_val = p.update(self.gbest_pos, self.fitness_f)

				if pbest_val < self.gbest_val:
					self.gbest_pos = pbest_pos.copy()
					self.gbest_val = float(pbest_val)
					improved = True

			if verbose:
				print(f"[Iter {it+1}/{num_iterations}] gbest_val = {self.gbest_val:.6g}")

			# Early stopping
			if tol is not None:
				if abs(best_val_prev - self.gbest_val) <= tol:
					no_improve += 1
				else:
					no_improve = 0
				best_val_prev = self.gbest_val

			if patience is not None and no_improve >= patience:
				if verbose:
					print(f"Early stop: no improvement > tol for {patience} iterations.")
				break

		return self.gbest_pos
