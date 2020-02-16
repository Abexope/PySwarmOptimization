"""测试用例2：单算法动态可视化"""

from optimizer.pso import Optimizer
from swarm.particle import *
from evaluator.base import *
from support.visualize import *


def pso_opt(D, pop_size, max_iter, pop_max, pop_min, V_max, V_min, fitness_function, weight=None, rec_step=50):
	from swarm.particle import ParticleSwarm
	
	swarm = ParticleSwarm(D, pop_size, pop_max, pop_min, V_max, V_min)  # 种群初始化
	opt = Optimizer(max_iter, swarm, fitness_function, rec_step=rec_step, is_record=True)  # 优化器初始化
	if weight:
		opt.fit(weight)  # 优化迭代
	else:
		opt.fit()

	AlgorithmVisual(opt.recoder, evaluator, pop_min_, pop_max_, is_save=True)


def qpso_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None, rec_step=50):
	
	swarm = QuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, fitness_function, rec_step=rec_step, is_record=True)
	if alpha:
		opt.fit(alpha)
	else:
		opt.fit()

	AlgorithmVisual(opt.recoder, evaluator, pop_min_, pop_max_, is_save=True)


def rqpso_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None, beta=None, rec_step=50):
	
	swarm = RevisedQuantumParticleSwarm(D, pop_size, pop_max, pop_min)      # 种群初始化
	opt = Optimizer(max_iter, swarm, evaluator=fitness_function, rec_step=rec_step, is_record=True)            # 优化算法初始化
	if alpha and beta:
		opt.fit(alpha, beta)
	else:
		opt.fit()           # 优化求解

	AlgorithmVisual(opt.recoder, evaluator, pop_min_, pop_max_, is_save=True)             # 结果可视化


if __name__ == '__main__':
	D_ = 2
	pop_size_ = 50
	max_iter_ = 5000
	pop_max_ = np.array([10 for _ in range(D_)])
	pop_min_ = np.array([-10 for _ in range(D_)])
	V_max_ = np.array([1 for _ in range(D_)])
	V_min_ = np.array([-1 for _ in range(D_)])
	evaluator = Griewank
	
	# pso_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, V_max_, V_min_, evaluator)        # 需要手动关闭当前动图才能显示下一个算法
	# qpso_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)
	rqpso_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)
