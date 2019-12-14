"""
Particle object module
"""

import numpy as np
from numpy.random import uniform, randn


class ParticleSwarm:
	
	def __init__(
			self, dimension: int, population: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray,
			upper_velocity: np.ndarray, lower_velocity: np.ndarray,
			acceleration_coeff1=2., acceleration_coeff2=2.,
	):
		self.D = dimension
		self.population = population
		self.upper = upper_bound
		self.lower = lower_bound
		self.V_max = upper_velocity
		self.V_min = lower_velocity
		self.c1 = acceleration_coeff1
		self.c2 = acceleration_coeff2
		
		self.position = self.lower + (self.upper - self.lower) * uniform(0, 1, size=[self.population, self.D])
		self.velocity = self.V_min + (self.V_max - self.V_min) * uniform(0, 1, size=[self.population, self.D])
		
		self.pbest = self.gbest = None
		
	def _correct_position(self):
		self.position = np.maximum(self.position, self.lower)
		self.position = np.minimum(self.position, self.upper)
		
	def _correct_velocity(self):
		self.velocity = np.maximum(self.velocity, self.V_min)
		self.velocity = np.minimum(self.velocity, self.V_max)
	
	def evolve(self):
		self.velocity = self.velocity \
			+ self.c1 * uniform(0, 1) * (self.pbest - self.position) \
			+ self.c2 * uniform(0, 1) * (self.gbest - self.position)
		self._correct_velocity()
		self.position = self.position + self.velocity
		self._correct_position()


class QuantumParticleSwarm:

	def __init__(
			self, dimension: int, population: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray,
	):
		self.D = dimension
		self.population = population
		self.upper = upper_bound
		self.lower = lower_bound

		self.position = self.lower + (self.upper - self.lower) * uniform(0, 1, size=[self.population, self.D])

		self.pbest = self.gbest = None

	def _correct_position(self):
		self.position = np.maximum(self.position, self.lower)
		self.position = np.minimum(self.position, self.upper)

	def evolve(self, alpha: float):
		mean_best = self.pbest.mean(axis=0)  # 平均最好位置
		phi = uniform(0, 1, size=(self.population, self.D))  # 收敛因子
		p = phi * self.pbest + (1 - phi) * self.gbest
		u = uniform(0, 1, size=(self.population, self.D))
		l = alpha * np.sign(u - 0.5) * np.abs(mean_best - self.position) * np.log(1 / u)
		self.position = p + l
		self._correct_position()


class RevisedQuantumParticleSwarm:

	def __init__(
			self, dimension: int, population: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray,
	):
		self.D = dimension
		self.population = population
		self.upper = upper_bound
		self.lower = lower_bound

		self.position = self.lower + (self.upper - self.lower) * uniform(0, 1, size=[self.population, self.D])

		self.pbest = self.gbest = None

	def _correct_position(self):
		self.position = np.maximum(self.position, self.lower)
		self.position = np.minimum(self.position, self.upper)

	def evolve(self, alpha: float, beta: float):
		mean_best = self.pbest.mean(axis=0)     # 平均最好位置
		phi = uniform(0, 1, size=(self.population, self.D))     # 收敛因子
		p = phi * self.pbest + (1 - phi) * self.gbest
		u1 = uniform(0, 1, size=(self.population, self.D))
		u2 = uniform(0, 1, size=(self.population, self.D))
		l1 = alpha * np.sign(u1 - 0.5) * np.abs(mean_best - self.position) * np.log(1 / u1)
		l2 = beta * np.sign(u2 - 0.5) * np.abs(p - self.position) * randn(self.population, self.D)
		self.position = p + 0.35 * l1 + 0.35 * l2
		self._correct_position()
