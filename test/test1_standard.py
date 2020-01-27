"""
测试用例1：算法的基本调用与收敛过程可视化
"""

from optimizer.pso import Optimizer
from evaluator.base import Sphere
from swarm.particle import *
from support.timer import Timer
import matplotlib.pyplot as plt


@Timer
def pso_opt(D, pop_size, max_iter, pop_max, pop_min, V_max, V_min, fitness_function, weight=None):
	
	swarm = ParticleSwarm(D, pop_size, pop_max, pop_min, V_max, V_min)      # 种群初始化
	opt = Optimizer(max_iter, swarm, fitness_function)                      # 优化器初始化
	if weight:
		opt.fit(weight)      # 优化迭代
	else:
		opt.fit()
	
	plt.figure(figsize=(5, 3))          # 结果可视化
	plt.loglog(range(max_iter // opt.rec_step), opt.recoder.fitness)
	plt.title('PSO')
	plt.grid()


@Timer
def qpso_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None):
	
	swarm = QuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, fitness_function)
	if alpha:
		opt.fit(alpha)
	else:
		opt.fit()
	
	plt.figure(figsize=(5, 3))
	plt.loglog(range(max_iter // opt.rec_step), opt.recoder.fitness)
	plt.title('QPSO')
	plt.grid()


@Timer
def rqpso_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None, beta=None):
	swarm = RevisedQuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, evaluator=fitness_function)
	if alpha and beta:
		opt.fit(alpha, beta)
	else:
		opt.fit()
	plt.figure(figsize=(5, 3))
	plt.loglog(range(max_iter // opt.rec_step), opt.recoder.fitness)
	plt.title('RQPSO')
	plt.grid()


if __name__ == '__main__':
	# 超参数配置
	D_ = 2
	pop_size_ = 50
	max_iter_ = 10000
	pop_max_ = np.array([10, 10])
	pop_min_ = np.array([-10, -10])
	V_max_ = np.array([1, 1])
	V_min_ = np.array([-1, -1])
	evaluator = Sphere

	pso_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, V_max_, V_min_, evaluator)
	qpso_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)
	rqpso_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)

	plt.show()
