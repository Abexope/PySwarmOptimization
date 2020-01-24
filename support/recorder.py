"""
优化过程记录
"""

import numpy as np
from abc import ABCMeta, abstractmethod


class BaseRecorder(metaclass=ABCMeta):
	"""迭代搜素信息记录"""
	
	@abstractmethod
	def __init__(self, alg_name: str, epoch: int, rec_step):
		self._name = alg_name
		self._epoch = epoch
		self._t = np.zeros(shape=(self.epoch,))
		self._doc = None
		self._rec_step = rec_step

	@property
	def name(self): return self._name

	@property
	def t(self): return self._t

	@property
	def epoch(self): return self._epoch
	
	@property
	def doc(self): return self._doc

	@property
	def step(self): return self._rec_step
	
	@abstractmethod
	def record(self, *args):
		pass


class PbestRecorder(BaseRecorder):
	"""个体历史最优记录"""
	
	def __init__(self, alg_name: str, epoch: int, population: int, dimension: int, rec_step: int):
		super(PbestRecorder, self).__init__(alg_name, epoch, rec_step)
		self._N = population  # 种群规模记录
		self._D = dimension  # 空间维度
		# self._t = np.zeros(shape=(self.epoch,))
		self._doc = np.zeros(shape=(self.epoch, self.N, self.D))  # 存储空间预分配
	
	@property
	def N(self): return self._N
	
	@property
	def D(self): return self._D
	
	def record(self, epc, rec):
		assert rec.shape == (self.N, self.D)  # 维数确认
		self._doc[epc // self.step, :, :] = rec  # 数据写入
		self._t[epc // self.step] = epc    # 迭代时间戳写入


class GbestRecorder(BaseRecorder):
	"""全局最优位置记录"""
	
	def __init__(self, alg_name: str, epoch: int, dimension: int, rec_step: int):
		super(GbestRecorder, self).__init__(alg_name, epoch, rec_step)
		self._D = dimension
		# self._t = np.zeros(shape=(self.epoch,))
		self._doc = np.zeros((self.epoch, self.D))
	
	@property
	def D(self): return self._D
	
	@property
	def doc(self): return self._doc
	
	def record(self, epc, rec):
		assert rec.shape == (self.D,)
		self._doc[epc // self.step, :] = rec
		self._t[epc // self.step] = epc


class FitnessRecorder(BaseRecorder):
	"""全局最优适应度值记录"""
	
	def __init__(self, alg_name: str, epoch: int, rec_step: int):
		super(FitnessRecorder, self).__init__(alg_name, epoch, rec_step)
		# self._t = np.zeros(shape=(self.epoch,))
		self._doc = np.zeros((self.epoch,))
	
	def record(self, epc, rec):
		self._doc[epc // self.step] = rec
		self._t[epc // self.step] = epc


class Recorder:
	
	def __init__(self, alg_name: str, epoch: int, N: int, D: int, rec_step: int):
		self.pbest_rec = PbestRecorder(alg_name, epoch, N, D, rec_step)
		self.gbest_rec = GbestRecorder(alg_name, epoch, D, rec_step)
		self.fitness_rec = FitnessRecorder(alg_name, epoch, rec_step)
		self._name = alg_name
		self._epoch = epoch
		self._N = N
		self._D = D
		self._rec_step = rec_step

	@property
	def name(self): return self._name

	@property
	def t(self): return self.fitness_rec.t

	@property
	def epoch(self): return self._epoch

	@property
	def N(self): return self._N

	@property
	def D(self): return self._D

	@property
	def rec_step(self): return self._rec_step
