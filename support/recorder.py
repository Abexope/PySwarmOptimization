"""
优化过程记录
"""

import numpy as np
from abc import ABCMeta, abstractmethod


class BaseRecorder(metaclass=ABCMeta):
	"""迭代搜素信息记录"""
	
	@abstractmethod
	def __init__(self, alg_name: str, epoch: int):
		self._name = alg_name
		self._epoch = epoch
		self._doc = None

	@property
	def name(self): return self._name

	@property
	def epoch(self): return self._epoch
	
	@property
	def doc(self): return self._doc
	
	@abstractmethod
	def record(self, *args):
		pass


class PbestRecorder(BaseRecorder):
	"""个体历史最优记录"""
	
	def __init__(self, alg_name: str, epoch: int, population: int, dimension: int):
		super(PbestRecorder, self).__init__(alg_name, epoch)
		self._N = population  # 种群规模记录
		self._D = dimension  # 空间维度
		self._doc = np.zeros(shape=(self.epoch, self.N, self.D))  # 存储空间预分配
	
	@property
	def N(self): return self._N
	
	@property
	def D(self): return self._D
	
	def record(self, epc, rec):
		assert rec.shape == (self.N, self.D)  # 维数确认
		self._doc[epc, :, :] = rec  # 数据写入


class GbestRecorder(BaseRecorder):
	"""全局最优位置记录"""
	
	def __init__(self, alg_name: str, epoch: int, dimension: int):
		super(GbestRecorder, self).__init__(alg_name, epoch)
		self._D = dimension
		self._doc = np.zeros((self.epoch, self.D))
	
	@property
	def D(self): return self._D
	
	@property
	def doc(self): return self._doc
	
	def record(self, epc, rec):
		assert rec.shape == (self.D,)
		self._doc[epc, :] = rec


class FitnessRecorder(BaseRecorder):
	"""全局最优适应度值记录"""
	
	def __init__(self, alg_name: str, epoch: int):
		super(FitnessRecorder, self).__init__(alg_name, epoch)
		self._doc = np.zeros((self.epoch,))
	
	def record(self, epc, rec):
		self._doc[epc] = rec


class Recorder:
	
	def __init__(self, alg_name: str, epoch: int, N: int, D: int):
		self.pbest_rec = PbestRecorder(alg_name, epoch, N, D)
		self.gbest_rec = GbestRecorder(alg_name, epoch, D)
		self.fitness_rec = FitnessRecorder(alg_name, epoch)
		self._name = alg_name
		self._epoch = epoch
		self._N = N
		self._D = D

	@property
	def name(self): return self._name

	@property
	def epoch(self): return self._epoch

	@property
	def N(self): return self._N

	@property
	def D(self): return self._D
