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
			evaluator, is_record=True, rec_step=50   # 需要满足 rec_step ≤ epoch 且能整除 epoch！！！
	):
		self._evaluator = evaluator
		self._epoch = epoch
		self._swarm = swarm
		self._rec_step = rec_step       # 记录步长，默认为1
		
		self.fitness = self._evaluator.infer(self.swarm.position)
		
		self.swarm.pbest = self.swarm.position
		self.swarm.gbest = self.swarm.position[np.argmin(self.fitness)]
		
		self.pbest_fitness = self.fitness
		self.gbest_fitness = np.min(self.fitness)

		self.is_record = is_record
		if is_record:
			self.recoder = Recorder(self.name, self._epoch // self.rec_step, self.swarm.population, self.swarm.D, self.rec_step)  # 迭代记录器

	@property
	def evaluator(self): return self._evaluator

	@property
	def swarm(self): return self._swarm

	@property
	def name(self): return self.swarm.name

	@property
	def rec_step(self): return self._rec_step

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
			if self.is_record and epc % self.rec_step == 0:
				self.recoder.pbest_rec.record(epc, self.swarm.pbest)
				self.recoder.gbest_rec.record(epc, self.swarm.gbest)
				self.recoder.fitness_rec.record(epc, self.gbest_fitness)
