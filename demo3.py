"""动态可视化"""

import numpy as np
from optimizer.pso import Optimizer
from evaluator.base import FitnessFunction2
import matplotlib.pyplot as plt
from support.visualize import *


def particle_swarm_opt(D, pop_size, max_iter, pop_max, pop_min, V_max, V_min, fitness_function, weight=None):
	from swarm.particle import ParticleSwarm
	
	swarm = ParticleSwarm(D, pop_size, pop_max, pop_min, V_max, V_min)  # 种群初始化
	opt = Optimizer(max_iter, swarm, fitness_function)  # 优化器初始化
	if weight:
		opt.fit(weight)  # 优化迭代
	else:
		opt.fit()
		
	return opt.recoder


def quantum_particle_swarm_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None):
	from swarm.particle import QuantumParticleSwarm
	
	swarm = QuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, fitness_function)
	if alpha:
		opt.fit(alpha)
	else:
		opt.fit()
	
	return opt.recoder


def revised_quantum_particle_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None, beta=None):
	from swarm.particle import RevisedQuantumParticleSwarm
	
	swarm = RevisedQuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, evaluator=fitness_function)
	if alpha and beta:
		opt.fit(alpha, beta)
	else:
		opt.fit()
	
	return opt.recoder


if __name__ == '__main__':
	D_ = 2
	pop_size_ = 50
	max_iter_ = 2000
	pop_max_ = np.array([10 for _ in range(D_)])
	pop_min_ = np.array([-10 for _ in range(D_)])
	V_max_ = np.array([1 for _ in range(D_)])
	V_min_ = np.array([-1 for _ in range(D_)])
	evaluator = FitnessFunction2
	
	pso_rec = particle_swarm_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, V_max_, V_min_, evaluator)
	qpso_rec = quantum_particle_swarm_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)
	rqpso_rec = revised_quantum_particle_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)
	
	GbestVisual(
		[pso_rec.gbest_rec, qpso_rec.gbest_rec, rqpso_rec.gbest_rec],
		[pso_rec.fitness_rec, qpso_rec.fitness_rec, rqpso_rec.fitness_rec]
	)
	# PbestVisual([pso_rec.pbest_rec, qpso_rec.pbest_rec, rqpso_rec.pbest_rec])
