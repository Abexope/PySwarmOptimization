"""
Particle object module
"""

from abc import abstractmethod, ABCMeta
import numpy as np
from numpy.random import uniform


class BaseParticle(metaclass=ABCMeta):
	"""
	Basic particle template
	"""
	
	@abstractmethod
	def __init__(self, dimension, upper_bound, lower_bound, *args):
		"""
		Initialization method
		:param dimension:
			space dimension
			python.int
		:param upper_bound:
			upper bound of searching space for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
		:param lower_bound:
			upper bound of searching space for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
		:param args:
			other inputs
		"""
		self.D = dimension
		self.upper = upper_bound
		self.lower = lower_bound
		self.__transform_pos_type()
	
	def __transform_pos_type(self):
		if not isinstance(self.upper, np.ndarray):
			self.upper = np.array(self.upper)
		if not isinstance(self.lower, np.ndarray):
			self.lower = np.array(self.lower)
		assert self.upper.ndim == self.lower.ndim == 1
		assert len(self.upper) == len(self.lower) == self.D

	@abstractmethod
	def evolve(self, *args):

		pass
	
	@abstractmethod
	def correct(self, *args):

		pass


class BaseSwarm(metaclass=ABCMeta):
	"""Basic swarm template"""

	@abstractmethod
	def __init__(
			self, dimension, population,
			# evaluator,
			upper_bound, lower_bound, *args
	):
		self.D = dimension
		self.population = population
		# self.evaluator = evaluator
		self.upper = upper_bound
		self.lower = lower_bound
		self.__transform_bound_type()
		self.position = self.lower + (self.upper - self.lower) \
			* uniform(0, 1, size=(self.population, self.D))     # position initialize

	def __transform_bound_type(self):
		if not isinstance(self.upper, np.ndarray):
			self.upper = np.array(self.upper)
		if not isinstance(self.lower, np.ndarray):
			self.lower = np.array(self.lower)
		assert self.upper.ndim == self.lower.ndim == 1
		assert len(self.upper) == len(self.lower) == self.D

	@abstractmethod
	def evolve(self, *args):

		pass


class Particle(BaseParticle):
	"""
	Particle object, the individual of basic PSO algorithm.
	"""
	
	def __init__(
			self, dimension, upper_bound, lower_bound,
			upper_velocity, lower_velocity,
			acceleration_coeff1=2, acceleration_coeff2=2,
	):
		"""
		Initialization method
		:param dimension:
			space dimension
			python.int
		:param upper_bound:
			upper bound of searching space for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
		:param lower_bound:
			lower bound of searching space for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
		:param upper_velocity:
			upper velocity of the particle for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
		:param lower_velocity:
			lower velocity of the particle for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
		:param acceleration_coeff1:
			acceleration coefficient of individual history impact.
			python.int
		:param acceleration_coeff2:
			acceleration coefficient of the best individual in the whole swarm.
			python.int
		"""
		super(Particle, self).__init__(dimension, upper_bound, lower_bound)
		self.V_max = upper_velocity
		self.V_min = lower_velocity
		self.__transform_vel_type()
		self.c1 = acceleration_coeff1
		self.c2 = acceleration_coeff2
		self.position = self.lower + (self.upper - self.lower) * uniform(0, 1, size=(self.D,))
		self.velocity = self.V_min + (self.V_max - self.V_min) * uniform(0, 1, size=(self.D,))
		self.pbest = self.position      # the personal history best position of a particle
		
	def __transform_vel_type(self):
		if not isinstance(self.V_max, np.ndarray):
			self.V_max = np.array(self.V_max)
		if not isinstance(self.V_min, np.ndarray):
			self.V_min = np.array(self.V_min)
		assert self.V_max.ndim == self.V_min.ndim == 1
		assert len(self.V_max) == len(self.V_min) == self.D
		
	def evolve(self, gbest, *args):
		"""
		Evolution method of basic particle in PSO algorithm
		:param gbest:
			position of the best individual in the whole swarm
			1-Axis numpy.array
		:param args:
			other params
		:return:
		"""
		self.velocity = self.velocity \
			+ self.c1 * uniform(0, 1) * (self.pbest - self.position) \
			+ self.c2 * uniform(0, 1) * (gbest - self.position)     # update the velocity
		self.position = self.position + self.velocity               # update the position
	
	def correct(self, *args):
		"""
		Correction method
			after once iteration, the values that are beyond the thresholds should be corrected.
		:param args:
		:return:
		"""
		self.position = np.minimum(self.position, self.upper)
		self.position = np.maximum(self.position, self.lower)
		self.velocity = np.minimum(self.velocity, self.V_max)
		self.velocity = np.maximum(self.velocity, self.V_min)


class ParticleSwarm(BaseSwarm):

	def __init__(
			self, dimension, population,
			# evaluator,
			upper_bound, lower_bound,
			upper_velocity, lower_velocity, acceleration_coeff1=2, acceleration_coeff2=2,
			opt_mode="minimum",
	):
		super(ParticleSwarm, self).__init__(
			dimension, population,
			# evaluator,
			upper_bound, lower_bound
		)
		assert opt_mode == "minimum" or "maximum"
		self.c1 = acceleration_coeff1
		self.c2 = acceleration_coeff2
		self.V_max = upper_velocity
		self.V_min = lower_velocity
		self.__transform_vel_type()
		self.velocity = self.V_min + (self.V_max - self.V_min) \
			* uniform(0, 1, size=(self.population, self.D))
		self.pbest = self.position      # initial personal best position

	def __transform_vel_type(self):
		if not isinstance(self.V_max, np.ndarray):
			self.V_max = np.array(self.V_max)
		if not isinstance(self.V_min, np.ndarray):
			self.V_min = np.array(self.V_min)
		assert self.V_max.ndim == self.V_min.ndim == 1
		assert len(self.V_max) == len(self.V_min) == self.D

	def evolve(self, gbest):
		# update velocity
		self.velocity = self.velocity \
			+ self.c1 * uniform(0, 1, size=(self.population, self.D)) * (self.pbest - self.position) \
			+ self.c2 * uniform(0, 1, size=(self.population, self.D)) * (gbest - self.position)
		self.__correct()
		self.position = self.position + self.velocity      # update position
		self.__correct()        # correct position

	def __correct(self):
		self.position = np.minimum(self.position, self.upper)
		self.position = np.maximum(self.position, self.lower)
		self.velocity = np.minimum(self.velocity, self.V_max)
		self.velocity = np.maximum(self.velocity, self.V_min)


if __name__ == '__main__':
	"""
	Debug in 9/27/2019: instantiation of object
	p = Particle(dimension=2, upper_bound=np.array([1, 1]), lower_bound=np.array([0, 0]))
	print(p.D, p.upper, p.lower)
	"""
	p = Particle(
		dimension=2, upper_bound=np.array([1, 1]), lower_bound=np.array([0, 0]),
		upper_velocity=np.array([-1, -1]), lower_velocity=np.array([1, 1]))
	print(p.D, p.upper, p.lower, p.position, p.velocity)
	pass
