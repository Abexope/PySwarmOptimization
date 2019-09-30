"""
test demo
"""

from swarm.particle import Particle
from swarm.particle import ParticleSwarm
from optimizer.pso import ParticleSwarmOptimization as PSO
from optimizer.pso import ParticleSwarmOptimization2 as PSO2
from evaluator.base import FitnessFunction
from evaluator.base import FitnessFunction2
import matplotlib.pyplot as plt


def demo_pso1():
	D = 2
	pop_size = 50
	max_iter = 100
	pop_max = [10, 10]
	pop_min = [-10, -10]
	V_max = [1, 1]
	V_min = [-1, -1]

	swarm = [Particle(D, pop_max, pop_min, V_max, V_min) for _ in range(pop_size)]
	pso = PSO(evaluator=FitnessFunction, swarm=swarm, population_size=pop_size)
	pso.search()

	plt.figure()
	plt.plot(range(max_iter), pso.yy)
	plt.grid()
	plt.ylim([0, 0.1])

	# plt.show()


def demo_pso2():
	D = 2
	pop_size = 50
	max_iter = 100
	pop_max = [10, 10]
	pop_min = [-10, -10]
	V_max = [1, 1]
	V_min = [-1, -1]
	swarm = ParticleSwarm(D, pop_size, pop_max, pop_min, V_max, V_min)
	pso = PSO2(epoch=max_iter, evaluator=FitnessFunction2, swarm=swarm)
	pso.search()

	plt.figure()
	plt.plot(range(max_iter), pso.yy)
	plt.grid()
	plt.ylim([0, 0.1])

	plt.show()


if __name__ == '__main__':
	# import time
	#
	# t1 = time.time()
	# for _ in range(100):
	# 	demo_pso1()
	# print("面向个体：", time.time() - t1)
	#
	# t2 = time.time()
	# for _ in range(100):
	# 	demo_pso2()
	# print("面向种群：", time.time() - t2)
	demo_pso1()
	demo_pso2()
