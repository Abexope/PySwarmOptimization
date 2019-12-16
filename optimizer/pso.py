"""
PSO and improvement algorithms are defined in this py-file
"""

from swarm.particle import *
import numpy as np


class Optimizer:
	
	def __init__(
			self, epoch: int,
			swarm: (ParticleSwarm, QuantumParticleSwarm, RevisedQuantumParticleSwarm),
			evaluator,
	):
		self.epoch = epoch
		self.evaluator = evaluator
		self.swarm = swarm
		
		self.fitness = self.evaluator.infer(self.swarm.position)
		
		self.swarm.pbest = self.swarm.position
		self.swarm.gbest = self.swarm.position[np.argmin(self.fitness)]
		
		self.pbest_fitness = self.fitness
		self.gbest_fitness = np.min(self.fitness)
		
		self.yy = []
	
	def _update_pbest(self):
		for j in range(self.swarm.population):
			self.pbest_fitness[j], self.swarm.pbest[j] = (self.fitness[j], self.swarm.position[j]) \
				if self.fitness[j] < self.pbest_fitness[j] \
				else (self.pbest_fitness[j], self.swarm.pbest[j])
	
	def _update_gbest(self):
		i = np.argmin(self.pbest_fitness)
		self.swarm.gbest = self.swarm.position[i]
		self.gbest_fitness = self.pbest_fitness[i]
	
	def search(self, *args):
		for epc in range(self.epoch):
			self.swarm.evolve(*args)                                    # swarm evolution
			self.fitness = self.evaluator.infer(self.swarm.position)    # update fitness value
			self._update_pbest()        # update personal best position and fitness value for each individual
			self._update_gbest()        # update global best position and fitness value
			self.yy.append(self.gbest_fitness)
