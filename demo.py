"""
test demo
"""

import numpy as np
from optimizer.pso import Optimizer
from evaluator.base import FitnessFunction2
import matplotlib.pyplot as plt


class Timer:
	"""Clock decorator"""
	
	def __init__(self, func):
		self.func = func
	
	def __call__(self, *args, **kwargs):
		from time import time
		t = time()
		self.func(*args, **kwargs)
		print("duration: {:.4f}".format(time() - t))


@Timer
def particle_swarm_opt(D, pop_size, max_iter, pop_max, pop_min, V_max, V_min, fitness_function, weight=None):
	from swarm.particle import ParticleSwarm
	
	swarm = ParticleSwarm(D, pop_size, pop_max, pop_min, V_max, V_min)      # 种群初始化
	opt = Optimizer(max_iter, swarm, fitness_function)                      # 优化器初始化
	if weight:
		opt.fit(weight)      # 优化迭代
	else:
		opt.fit()
	
	plt.figure(figsize=(5, 3))          # 结果可视化
	plt.loglog(range(max_iter), opt.yy)
	plt.title('PSO')
	plt.grid()


@Timer
def quantum_particle_swarm_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None):
	from swarm.particle import QuantumParticleSwarm
	
	swarm = QuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, fitness_function)
	if alpha:
		opt.fit(alpha)
	else:
		opt.fit()
	
	plt.figure(figsize=(5, 3))
	plt.loglog(range(max_iter), opt.yy)
	plt.title('QPSO')
	plt.grid()


@Timer
def revised_quantum_particle_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None, beta=None):
	from swarm.particle import RevisedQuantumParticleSwarm
	
	swarm = RevisedQuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, evaluator=fitness_function)
	if alpha and beta:
		opt.fit(alpha, beta)
	else:
		opt.fit()
	plt.figure(figsize=(5, 3))
	plt.loglog(range(max_iter), opt.yy)
	plt.title('RQPSO')
	plt.grid()


if __name__ == '__main__':
	# 超参数配置
	D_ = 2
	pop_size_ = 50
	max_iter_ = 100000
	pop_max_ = np.array([10, 10])
	pop_min_ = np.array([-10, -10])
	V_max_ = np.array([1, 1])
	V_min_ = np.array([-1, -1])
	evaluator = FitnessFunction2

	particle_swarm_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, V_max_, V_min_, evaluator)
	quantum_particle_swarm_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)
	revised_quantum_particle_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)

	plt.show()
