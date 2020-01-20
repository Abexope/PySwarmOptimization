"""
Particle object module
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.random import uniform, randn


class Swarm(metaclass=ABCMeta):
	name = None
	
	@abstractmethod
	def __init__(
			self, dimension: int, population: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray,
	):
		self._dimension = dimension
		self._population = population
		self._upper = upper_bound
		self._lower = lower_bound
		self.pbest = self.gbest = None
	
	@property
	def D(self): return self._dimension
	
	@property
	def population(self): return self._population
	
	@property
	def upper(self): return self._upper
	
	@property
	def lower(self): return self._lower
	
	@abstractmethod
	def evolve(self, *args):
		pass
	
	@abstractmethod
	def _correct_position(self, *args):
		pass


class ParticleSwarm(Swarm):
	name = "PSO"
	
	def __init__(
			self, dimension: int, population: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray,
			upper_velocity: np.ndarray, lower_velocity: np.ndarray,
			acceleration_coeff1=2., acceleration_coeff2=2.,
	):
		super(ParticleSwarm, self).__init__(dimension, population, upper_bound, lower_bound)
		self._Vmax = upper_velocity
		self._Vmin = lower_velocity
		self._c1 = acceleration_coeff1
		self._c2 = acceleration_coeff2
		self._position = self.lower + (self.upper - self.lower) * uniform(0, 1, size=[self.population, self.D])
		self._velocity = self.V_min + (self.V_max - self.V_min) * uniform(0, 1, size=[self.population, self.D])
		
	@property
	def position(self): return self._position
	
	@property
	def velocity(self): return self._velocity
	
	@property
	def c1(self): return self._c1
	
	@property
	def c2(self): return self._c2
	
	@property
	def V_max(self): return self._Vmax
	
	@property
	def V_min(self): return self._Vmin
		
	def _correct_position(self):
		self._position = np.maximum(self.position, self.lower)
		self._position = np.minimum(self.position, self.upper)
		
	def _correct_velocity(self):
		self._velocity = np.maximum(self.velocity, self.V_min)
		self._velocity = np.minimum(self.velocity, self.V_max)
	
	def evolve(self, weight=1.):
		self._velocity = self.velocity \
			+ self.c1 * uniform(0, 1) * (self.pbest - self.position) \
			+ self.c2 * uniform(0, 1) * (self.gbest - self.position)
		self._correct_velocity()
		self._position = self.position + weight * self.velocity
		self._correct_position()


class QuantumParticleSwarm(Swarm):
	name = "QPSO"

	def __init__(
			self, dimension: int, population: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray,
	):
		super(QuantumParticleSwarm, self).__init__(dimension, population, upper_bound, lower_bound)
		self._position = self.lower + (self.upper - self.lower) * uniform(0, 1, size=[self.population, self.D])

	@property
	def position(self): return self._position

	def _correct_position(self):
		self._position = np.maximum(self.position, self.lower)
		self._position = np.minimum(self.position, self.upper)

	def evolve(self, alpha=1.):
		mean_best = self.pbest.mean(axis=0)  # 平均最好位置
		phi = uniform(0, 1, size=(self.population, self.D))  # 收敛因子
		p = phi * self.pbest + (1 - phi) * self.gbest
		u = uniform(0, 1, size=(self.population, self.D))
		l1 = np.sign(u - 0.5) * np.abs(mean_best - self.position) * np.log(1 / u)
		self._position = p + alpha * l1
		self._correct_position()


class RevisedQuantumParticleSwarm(Swarm):
	name = "RQPSO"

	def __init__(
			self, dimension: int, population: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray,
	):
		super(RevisedQuantumParticleSwarm, self).__init__(dimension, population, upper_bound, lower_bound)
		self._position = self.lower + (self.upper - self.lower) * uniform(0, 1, size=[self.population, self.D])
	
	@property
	def position(self): return self._position
	
	def _correct_position(self):
		self._position = np.maximum(self.position, self.lower)
		self._position = np.minimum(self.position, self.upper)

	def evolve(self, alpha=.4, beta=.4):
		mean_best = self.pbest.mean(axis=0)     # 平均最好位置
		phi = uniform(0, 1, size=(self.population, self.D))     # 收敛因子
		p = phi * self.pbest + (1 - phi) * self.gbest
		u1 = uniform(0, 1, size=(self.population, self.D))
		u2 = uniform(0, 1, size=(self.population, self.D))
		l1 = np.sign(u1 - 0.5) * np.abs(mean_best - self.position) * np.log(1 / u1)
		l2 = np.sign(u2 - 0.5) * np.abs(p - self.position) * randn(self.population, self.D)
		self._position = p + alpha * l1 + beta * l2
		self._correct_position()
