"""
PSO and improvement algorithms are defined in this py-file
"""

from swarm.particle import ParticleSwarm, QuantumParticleSwarm, RevisedQuantumParticleSwarm
import numpy as np


class ParticleSwarmOptimization:

	def __init__(
			self, dimension: int, population_size: int, epoch: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray, upper_velocity: np.ndarray, lower_velocity: np.ndarray,
			evaluator=None,
	) -> None:
		
		self.D = dimension
		self.population = population_size
		self.epoch = epoch
		self.evaluator = evaluator
		self.upper, self.lower = upper_bound, lower_bound
		self.V_max, self.V_min = upper_velocity, lower_velocity
		
		self.swarm = ParticleSwarm(self.D, self.population, self.upper, self.lower, self.V_max, self.V_min)
		self.fitness = self.evaluator.infer(self.swarm.position)
		
		self.swarm.pbest = self.swarm.position
		self.swarm.gbest = self.swarm.position[np.argmin(self.fitness)]
		
		self.pbest_fitness = self.fitness
		self.gbest_fitness = np.min(self.fitness)
		
		self.yy = []
		
	def _update_pbest(self):
		for j in range(self.population):
			self.pbest_fitness[j], self.swarm.pbest[j] = (self.fitness[j], self.swarm.position[j]) \
				if self.fitness[j] < self.pbest_fitness[j] \
				else (self.pbest_fitness[j], self.swarm.pbest[j])

	def _update_gbest(self):
		self.swarm.gbest = self.swarm.position[np.argmin(self.pbest_fitness)]
		self.gbest_fitness = self.pbest_fitness[np.argmin(self.pbest_fitness)]

	def search(self):
		for epc in range(self.epoch):
			self.swarm.evolve()  # swarm evolution
			self.fitness = self.evaluator.infer(self.swarm.position)  # update fitness value
			self._update_pbest()  # update personal best position and fitness value for each individual
			self._update_gbest()  # update global best position and fitness value
			self.yy.append(self.gbest_fitness)


class QuantumParticleSwarmOptimization:

	def __init__(
			self, dimension: int, population_size: int, epoch: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray, evaluator=None,
	):
		self.D = dimension
		self.population = population_size
		self.epoch = epoch
		self.evaluator = evaluator
		self.upper, self.lower = upper_bound, lower_bound

		self.swarm = QuantumParticleSwarm(self.D, self.population, upper_bound, lower_bound)
		self.fitness = self.evaluator.infer(self.swarm.position)

		self.swarm.pbest = self.swarm.position
		self.swarm.gbest = self.swarm.position[np.argmin(self.fitness)]

		self.pbest_fitness = self.fitness
		self.gbest_fitness = np.min(self.fitness)

		self.yy = []

	def _update_pbest(self):
		for j in range(self.population):
			self.pbest_fitness[j], self.swarm.pbest[j] = (self.fitness[j], self.swarm.position[j]) \
				if self.fitness[j] < self.pbest_fitness[j] \
				else (self.pbest_fitness[j], self.swarm.pbest[j])

	def _update_gbest(self):
		self.swarm.gbest = self.swarm.position[np.argmin(self.pbest_fitness)]
		self.gbest_fitness = self.pbest_fitness[np.argmin(self.pbest_fitness)]

	def search(self):
		for epc in range(self.epoch):
			self.swarm.evolve(alpha=1)  # swarm evolution
			self.fitness = self.evaluator.infer(self.swarm.position)  # update fitness value
			self._update_pbest()  # update personal best position and fitness value for each individual
			self._update_gbest()  # update global best position and fitness value
			self.yy.append(self.gbest_fitness)
		pass


class RevisedQuantumParticleSwarmOptimization:

	def __init__(
			self, dimension: int, population_size: int, epoch: int,
			upper_bound: np.ndarray, lower_bound: np.ndarray, evaluator=None,
	):
		self.D = dimension
		self.population = population_size
		self.epoch = epoch
		self.evaluator = evaluator
		self.upper, self.lower = upper_bound, lower_bound

		self.swarm = RevisedQuantumParticleSwarm(self.D, self.population, upper_bound, lower_bound)
		self.fitness = self.evaluator.infer(self.swarm.position)

		self.swarm.pbest = self.swarm.position
		self.swarm.gbest = self.swarm.position[np.argmin(self.fitness)]

		self.pbest_fitness = self.fitness
		self.gbest_fitness = np.min(self.fitness)

		self.yy = []

	def _update_pbest(self):
		for j in range(self.population):
			self.pbest_fitness[j], self.swarm.pbest[j] = (self.fitness[j], self.swarm.position[j]) \
				if self.fitness[j] < self.pbest_fitness[j] \
				else (self.pbest_fitness[j], self.swarm.pbest[j])

	def _update_gbest(self):
		self.swarm.gbest = self.swarm.position[np.argmin(self.pbest_fitness)]
		self.gbest_fitness = self.pbest_fitness[np.argmin(self.pbest_fitness)]

	def search(self):
		for epc in range(self.epoch):
			self.swarm.evolve(alpha=1, beta=1)  # swarm evolution
			self.fitness = self.evaluator.infer(self.swarm.position)  # update fitness value
			self._update_pbest()  # update personal best position and fitness value for each individual
			self._update_gbest()  # update global best position and fitness value
			self.yy.append(self.gbest_fitness)
		pass
