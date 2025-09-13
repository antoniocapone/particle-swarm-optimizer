import numpy as np
from typing import Callable, Tuple, Optional


class Particle:
	def __init__(
		self,
		domain_dim: int,
		bounds: np.ndarray,
		inertia_weight: float,
		cognitive_coeff: float,
		social_coeff: float,
		v_max_frac: float = 0.3,
		rng: Optional[np.random.Generator] = None,
		fitness_f: Optional[Callable[[np.ndarray], float]] = None,
	) -> None:
		if rng is None:
			rng = np.random.default_rng()

		self.D = domain_dim
		self.bounds = np.asarray(bounds, dtype=float)
		assert self.bounds.shape == (self.D, 2), "bounds must be (D, 2)"

		self.w = float(inertia_weight)
		self.c1 = float(cognitive_coeff)
		self.c2 = float(social_coeff)

		self.range = self.bounds[:, 1] - self.bounds[:, 0]
		self.v_max = np.abs(v_max_frac) * self.range

		self.position = rng.random(self.D) * self.range + self.bounds[:, 0]

		self.velocity = (rng.random(self.D) * 2.0 - 1.0) * self.v_max

		self.pbest_pos = self.position.copy()
		if fitness_f is not None:
			self.pbest_val = float(fitness_f(self.position))
		else:
			self.pbest_val = np.inf

		self.rng = rng

	def _clamp_position(self) -> None:
		# Clamp duro nei bounds (puoi sostituire con “reflection” se preferisci)
		self.position = np.minimum(np.maximum(self.position, self.bounds[:, 0]), self.bounds[:, 1])

	def _clamp_velocity(self) -> None:
		self.velocity = np.minimum(np.maximum(self.velocity, -self.v_max), self.v_max)

	def get_pbest_pos(self) -> np.ndarray:
		return self.pbest_pos

	def get_pbest_val(self) -> float:
		return self.pbest_val

	def update(self, gbest_pos: np.ndarray, fitness_f: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float]:
		r1 = self.rng.random(self.D)
		r2 = self.rng.random(self.D)

		cognitive = self.c1 * r1 * (self.pbest_pos - self.position)
		social = self.c2 * r2 * (gbest_pos      - self.position)
		self.velocity = self.w * self.velocity + cognitive + social

		self._clamp_velocity()

		self.position = self.position + self.velocity
		self._clamp_position()

		f_val = float(fitness_f(self.position))

		if f_val < self.pbest_val:
			self.pbest_pos = self.position.copy()
			self.pbest_val = f_val

		return self.pbest_pos, self.pbest_val
