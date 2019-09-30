"""
PSO and improvement algorithms are defined in this py-file
"""

from abc import ABCMeta, abstractmethod
from swarm.particle import Particle
from swarm.particle import ParticleSwarm
import numpy as np


class BasePSO(metaclass=ABCMeta):
	"""
	Basic PSO template
	"""
	
	@abstractmethod
	def __init__(self, epoch, population_size, evaluator=None, opt_mode=None, *args):
		"""
		Initialization method
		:param epoch: 
			the maximum iteration numbers of the algorithm
			python.int
		:param evaluator: 
			objective function, it usually could be called at fitness function
			evaluator object
		:param opt_mode: 
			optimization mode
			python.string, just support "maximum" and "minimum"
		:param args: 
			other params
		"""
		self.epoch = epoch
		self.evaluator = evaluator
		self.population_size = population_size
		self.opt_mode = opt_mode
	
	@abstractmethod
	def search(self, *args):
		"""
		Definition of the algorithm flow
		:param args: 
		:return: 
		"""
		pass


class ParticleSwarmOptimization(BasePSO):
	
	def __init__(
			self, epoch=100, evaluator=None, opt_mode="minimum",
			swarm=None, dimension=None, population_size=None,
			upper_bound=None, lower_bound=None, upper_velocity=None, lower_velocity=None,
	):
		"""
		Initialization method
		:param epoch:
			the maximum iteration numbers of the algorithm
			python.int
		:param population_size:
			the number of all individuals in the whole swarm
			python.int
		:param evaluator:
			objective function, it usually could be called at fitness function
			evaluator object
		:param opt_mode:
			optimization mode
			python.string, just support "maximum" and "minimum"
		:param swarm:
			python.list
			particle swarm object, each element in list is a Particle object
		:param dimension:
			python.int
			space dimension, it must be defined if swarm is None
		:param upper_bound:
			upper bound of searching space for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
			it must be defined if swarm is None
		:param lower_bound:
			lower bound of searching space for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
			it must be defined if swarm is None
		:param upper_velocity:
			upper velocity of the particle for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
			it must be defined if swarm is None
		:param lower_velocity:
			lower velocity of the particle for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
			it must be defined if swarm is None
		"""
		super(ParticleSwarmOptimization, self).__init__(epoch, population_size)
		assert opt_mode == "maximum" or "minimum"
		self.evaluator = evaluator
		self.opt_mode = opt_mode
		self.population_size = population_size
		if swarm is None:
			if upper_bound is None or lower_bound is None or \
				upper_velocity is None or lower_velocity is None or dimension is None:
				raise ValueError(
					"the space dimension, population size, the boundary of position and velocity should be defined "
					"if swarm object is not transferred")
			self.D = dimension
			self.swarm = [
				Particle(
					dimension=self.D,
					upper_bound=upper_bound, lower_bound=lower_bound,
					upper_velocity=upper_velocity, lower_velocity=lower_velocity,
				) for _ in range(self.population_size)
			]   # swarm initialization
		else:
			self.swarm = swarm      # self-define swarm are supported
		self.fitness_value = [self.evaluator.infer(self.swarm[i].position) for i in range(self.population_size)]     # initial fitness value
		self.pbest_fitness = self.fitness_value     # initial personal history best fitness value
		if self.opt_mode == "minimum":
			self.gbest_fitness = min(self.pbest_fitness)    # initial global best fitness value
			self.gbest_individual = self.swarm[self.pbest_fitness.index(min(self.pbest_fitness))]   # initial global best individual
		else:
			self.gbest_fitness = max(self.pbest_fitness)  # initial global best fitness value
			self.gbest_individual = self.swarm[self.pbest_fitness.index(max(self.pbest_fitness))]  # initial global best individual
		self.yy = []
	
	def search(self):
		"""Definition of the algorithm flow"""
		for epc in range(self.epoch):
			for particle in self.swarm:
				particle.evolve(self.gbest_individual.position)     # swarm evolution
				particle.correct()      # position and velocity correction
			self.fitness_value = [self.evaluator.infer(particle.position) for particle in self.swarm]   # update fitness value
			if self.opt_mode == "minimum":
				for i, particle in enumerate(self.swarm):
					if self.fitness_value[i] < self.pbest_fitness[i]:
						self.pbest_fitness[i] = self.fitness_value[i]         # update the personal hist best fitness value
						self.swarm[i].pbest = self.swarm[i].position    # update the personal history best position
				self.gbest_fitness = min(self.pbest_fitness)
				self.gbest_individual = self.swarm[self.pbest_fitness.index(min(self.pbest_fitness))]
			else:
				for i, particle in enumerate(self.swarm):
					if self.fitness_value[i] > self.pbest_fitness[i]:
						self.pbest_fitness[i] = self.fitness_value[i]         # update the personal hist best fitness value
						self.swarm[i].pbest = self.swarm[i].position    # update the personal history best position
				self.gbest_fitness = max(self.pbest_fitness)
				self.gbest_individual = self.swarm[self.pbest_fitness.index(max(self.pbest_fitness))]
			self.yy.append(self.gbest_fitness)


class ParticleSwarmOptimization2(BasePSO):

	def __init__(
			self, epoch=100, evaluator=None, opt_mode="minimum",
			swarm=None, dimension=None, population_size=None,
			upper_bound=None, lower_bound=None, upper_velocity=None, lower_velocity=None,
	):
		super(ParticleSwarmOptimization2, self).__init__(epoch, population_size, evaluator, opt_mode)
		assert opt_mode == "minimum" or "maximum"
		self.opt_mode = opt_mode

		self.upper = upper_bound
		self.lower = lower_bound
		self.V_max = upper_velocity
		self.V_min = lower_velocity
		if swarm is None:
			self.D = dimension
			self.population_size = population_size
			self.swarm = ParticleSwarm(
				self.D, self.population_size, self.upper, self.lower,
				self.V_max, self.V_min, opt_mode=opt_mode
			)
		else:
			self.swarm = swarm      # outer swarm definition is supported

		self.fitness = self.get_fitness_value()     # initial fitness value of all individuals
		self.pbest_fitness = self.fitness           # initial personal best fitness value

		if self.opt_mode == "minimum":
			self.gbest = self.swarm.position[np.argmin(self.pbest_fitness)]
			self.gbest_fitness = self.pbest_fitness[np.argmin(self.pbest_fitness)]      # for robustness
		else:
			self.gbest = self.swarm.position[np.argmax(self.pbest_fitness)]
			self.gbest_fitness = self.pbest_fitness[np.argmax(self.pbest_fitness)]      # for robustness
		self.yy = []

	def get_fitness_value(self, fun=None):
		if fun is None:
			return self.evaluator.infer(self.swarm.position)
		else:
			return fun(self.swarm.position)

	def __update_pbest(self):
		if self.opt_mode == "minimum":
			self.pbest_fitness = np.minimum(self.fitness, self.pbest_fitness)
			self.swarm.pbest[self.fitness < self.pbest_fitness] = self.swarm.position[self.fitness < self.pbest_fitness]
		else:
			self.pbest_fitness = np.maximum(self.fitness, self.pbest_fitness)
			self.swarm.pbest[self.fitness > self.pbest_fitness] = self.swarm.position[self.fitness > self.pbest_fitness]

	def __update_gbest(self):
		if self.opt_mode == "minimum":
			self.gbest = self.swarm.position[np.argmin(self.pbest_fitness)]
			self.gbest_fitness = self.pbest_fitness[np.argmin(self.pbest_fitness)]
		else:
			self.gbest = self.swarm.position[np.argmax(self.pbest_fitness)]
			self.gbest_fitness = self.pbest_fitness[np.argmax(self.pbest_fitness)]

	def search(self, *args):
		"""Definition of algorithm flow"""
		for epc in range(self.epoch):
			self.swarm.evolve(self.gbest)               # swarm evolution
			self.fitness = self.get_fitness_value()     # update fitness value
			self.__update_pbest()   # update personal best position and fitness value for each individual
			self.__update_gbest()   # update global best position and fitness value
			self.yy.append(self.gbest_fitness)


if __name__ == '__main__':
	"""
	Debug at 09/30/2019
	opt = ParticleSwarmOptimization(
		dimension=2, epoch=100,
		)
	print(opt)
	"""
	
	pass
