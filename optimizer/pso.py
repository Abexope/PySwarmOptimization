"""
PSO and improvement algorithms are defined in this py-file
"""

from swarm.particle import *
from support.recorder import Recorder
import numpy as np


class Optimizer:
	
	def __init__(
			self, epoch: int,
			swarm: (ParticleSwarm, QuantumParticleSwarm, RevisedQuantumParticleSwarm),
			evaluator, is_record=True,
	):
		self._evaluator = evaluator
		self._epoch = epoch
		self._swarm = swarm
		self._name = swarm.name
		
		self.fitness = self._evaluator.infer(self.swarm.position)
		
		self.swarm.pbest = self.swarm.position
		self.swarm.gbest = self.swarm.position[np.argmin(self.fitness)]
		
		self.pbest_fitness = self.fitness
		self.gbest_fitness = np.min(self.fitness)
		
		self.yy = []
		self.is_record = is_record
		if is_record:
			self.recoder = Recorder(self.name, self._epoch, self.swarm.population, self.swarm.D)  # 迭代记录器

	@property
	def evaluator(self): return self._evaluator

	@property
	def swarm(self): return self._swarm

	@property
	def name(self): return self._name

	def _update_pbest(self):
		i = np.where(self.fitness < self.pbest_fitness)
		self.swarm.pbest[i] = self.swarm.position[i]
		self.pbest_fitness[i] = self.fitness[i]
	
	def _update_gbest(self):
		i = np.argmin(self.pbest_fitness)
		self.swarm.gbest = self.swarm.position[i]
		self.gbest_fitness = self.pbest_fitness[i]
	
	def fit(self, *args):
		for epc in range(self._epoch):
			self.swarm.evolve(*args)                                    # swarm evolution
			self.fitness = self._evaluator.infer(self.swarm.position)    # update fitness value
			self._update_pbest()        # update personal best position and fitness value for each individual
			self._update_gbest()        # update global best position and fitness value

			"""迭代记录接口"""
			if self.is_record:
				self.recoder.pbest_rec.record(epc, self.swarm.pbest)
				self.recoder.gbest_rec.record(epc, self.swarm.gbest)
				self.recoder.fitness_rec.record(epc, self.gbest_fitness)
			self.yy.append(self.gbest_fitness)
