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
