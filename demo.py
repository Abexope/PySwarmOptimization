"""
test demo
"""

import numpy as np
from optimizer.pso import ParticleSwarmOptimization as PSO
from evaluator.base import FitnessFunction2
import matplotlib.pyplot as plt


def demo_pso1():
	D = 2
	pop_size = 50
	max_iter = 100000
	pop_max = np.array([10, 10])
	pop_min = np.array([-10, -10])
	V_max = np.array([1, 1])
	V_min = np.array([-1, -1])

	pso = PSO(D, pop_size, max_iter, pop_max, pop_min, V_max, V_min, evaluator=FitnessFunction2)
	pso.search()

	plt.figure(figsize=(4, 3))
	plt.loglog(range(max_iter), pso.yy)
	plt.grid()
	# plt.ylim([1e-10, 0.1])

	plt.show()


if __name__ == '__main__':
	# import time
	#
	# t1 = time.time()
	# for _ in range(100):
	# 	demo_pso1()
	# print((time.time() - t1) / 100)

	demo_pso1()
