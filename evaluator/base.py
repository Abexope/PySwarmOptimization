"""
Evaluator module
"""

import numpy as np


class FitnessFunction:
	
	# def __init__(self, dimension):
	# 	self.D = dimension
	
	@staticmethod
	def infer(x):
		return np.sum(x ** 2)


class FitnessFunction2:

	# def __init__(self, dimension):
	# 	self.D = dimension

	@staticmethod
	def infer(x):
		return np.sum(x ** 2, axis=1)
