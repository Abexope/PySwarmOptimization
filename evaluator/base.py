"""
Evaluator module
"""

from abc import ABCMeta, abstractmethod
import numpy as np


class FitnessFunction(metaclass=ABCMeta):

	func_name = None

	@staticmethod
	@abstractmethod
	def infer(x):
		pass


class Sphere(FitnessFunction):

	func_name = "Sphere"

	def __new__(cls, *args, **kwargs):
		if not hasattr(cls, 'instance'):
			cls.instance = super(Sphere, cls).__new__(cls)     # 单例
		return cls.instance

	@staticmethod
	def infer(x):
		return np.sum(x ** 2, axis=1)


class Rastrigrin(FitnessFunction):

	func_name = "Rastrigrin"

	def __new__(cls, *args, **kwargs):
		if not hasattr(cls, 'instance'):
			cls.instance = super(Rastrigrin, cls).__new__(cls)     # 单例
		return cls.instance

	@staticmethod
	def infer(x):
		return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10, axis=1)


class Rosenbrock(FitnessFunction):

	func_name = "Rosenbrock"

	def __new__(cls, *args, **kwargs):
		if not hasattr(cls, 'instance'):
			cls.instance = super(Rosenbrock, cls).__new__(cls)     # 单例
		return cls.instance

	@staticmethod
	def infer(x):
		D = x.shape[1]      # 变量x的维数
		return np.sum((1 - x[:, :D - 1]) ** 2 + 100 * (x[:, 1:] - x[:, :D - 1] ** 2) ** 2, axis=1)


class Griewank(FitnessFunction):

	func_name = "Griewank"

	def __new__(cls, *args, **kwargs):
		if not hasattr(cls, 'instance'):
			cls.instance = super(Griewank, cls).__new__(cls)     # 单例
		return cls.instance

	@staticmethod
	def infer(x):
		D = x.shape[1]
		return np.sum(x ** 2, axis=1) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, D + 1))), axis=1) + 1


class Schaffer(FitnessFunction):

	func_name = "Schaffer"

	def __new__(cls, *args, **kwargs):
		if not hasattr(cls, 'instance'):
			cls.instance = super(Schaffer, cls).__new__(cls)     # 单例
		return cls.instance

	@staticmethod
	def infer(x):
		return 0.5 + (np.sin(np.sum(x ** 2, axis=1)) ** 2 - 0.5) / (1 + 0.001 * np.sum(x ** 2, axis=1)) ** 2


FUNCTIONS = (Sphere, Rastrigrin, Rosenbrock, Griewank, Schaffer)
