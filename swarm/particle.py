"""
Particle object module
"""

from abc import abstractmethod, ABCMeta
from numpy import array, ndarray
# from pandas import Series
# import numpy as np


class BasicParticle(metaclass=ABCMeta):
	"""
	Template of particle class
	"""
	
	@abstractmethod
	def __init__(self, dimension, upper_bound, lower_bound, *args):
		"""
		Initialization method
		:param dimension:
			space dimension
			python.int
		:param upper_bound:
			upper bound of searching space for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
		:param lower_bound:
			upper bound of searching space for each dimension
			1-Axis array like, only python list, numpy.array and pandas.Series are supported.
			Any data structure is always transformed to numpy.array
		:param args:
			other inputs
		"""
		self.D = dimension
		self.upper = upper_bound
		self.lower = lower_bound
		self.__transform_type()
	
	def __transform_type(self):
		if isinstance(self.upper, ndarray) and isinstance(self.lower, ndarray):
			return
		self.upper = array(self.upper)
		self.lower = array(self.upper)

	@abstractmethod
	def evolve(self, *args):
		
		pass
	
	@abstractmethod
	def correct(self, *args):
		
		pass


class Particle(BasicParticle):
	
	def __init__(self, dimension, upper_bound, lower_bound):
		super(Particle, self).__init__(dimension, upper_bound, lower_bound)
		pass
	
	def evolve(self, *args):
		pass
	
	def correct(self, *args):
		pass


if __name__ == '__main__':
	"""
	Debug in 9/27/2019: instantiation of object
	p = Particle(dimension=2, upper_bound=array([1, 1]), lower_bound=array([0, 0]))
	print(p.D, p.upper, p.lower)
	"""
	
	pass
