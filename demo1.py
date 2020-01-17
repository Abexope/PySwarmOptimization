"""
基础demo
"""

import numpy as np
from optimizer.pso import Optimizer
from evaluator.base import FitnessFunction2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')


class GbestVisual:
	"""全局最优的收敛过程可视化"""
	
	def __init__(self, loc_recoder, fit_recoder):
		self._loc_recoder = loc_recoder      # 位置记录
		self._fit_recoder = fit_recoder      # 适应度值记录
		self.epoch = loc_recoder.epoch      # 迭代次数
		self.D = loc_recoder.D              # 维度
		self._plot_canvas()                 # 初始化画布
	
	def _plot_canvas(self):
		self._fig = plt.figure()
		
		self.fit_ax = self._fig.add_subplot(121)     # 可视化最优适应度值收敛过程
		self.fit_ln, = self.fit_ax.semilogy([], [], linestyle='-', color='b', animated=False, label='fitness value')
		
		self.loc_ax = self._fig.add_subplot(122)     # 可视化最优位置收敛过程
		self.loc_ln1, = self.loc_ax.semilogy([], [], linestyle='-', animated=False, label='D1')
		self.loc_ln2, = self.loc_ax.semilogy([], [], linestyle='-', animated=False, label='D2')
		
		return self.fit_ln, self.loc_ln1, self.loc_ln2
		
	def _init_func(self):
		self.fit_ax.set_xlim(0, self.epoch)
		self.fit_ax.set_ylim(1e-310, 1e2)
		self.loc_ax.set_xlim(0, self.epoch)
		self.loc_ax.set_ylim(1e-310, 1e2)
		return self.fit_ln, self.loc_ln1, self.loc_ln2
		
	def _update(self, epc):
		self.fit_ln.set_data(range(epc), self._fit_recoder.doc[:epc])   # 画fitness
		self.loc_ln1.set_data(range(epc), self._loc_recoder.doc[:epc, 0])       # 画维度1
		self.loc_ln2.set_data(range(epc), self._loc_recoder.doc[:epc, 1])       # 画维度2
		return self.fit_ln, self.loc_ln1, self.loc_ln2
	
	def visual(self):
		_ani = FuncAnimation(
			self._fig, self._update, frames=self.epoch,
			init_func=self._init_func, blit=False, repeat=False, interval=0,
		)
		
	
class SearchVisual:
	"""种群搜索过程可视化，仅在维度为 2 时可用"""
	
	def __init__(self):
		
		pass


class Timer:
	"""Clock decorator"""
	
	def __init__(self, func):
		self.func = func
	
	def __call__(self, *args, **kwargs):
		from time import time
		t = time()
		self.func(*args, **kwargs)
		print("duration: {:.4f}".format(time() - t))


# @Timer
def particle_swarm_opt(D, pop_size, max_iter, pop_max, pop_min, V_max, V_min, fitness_function, weight=None):
	from swarm.particle import ParticleSwarm
	
	swarm = ParticleSwarm(D, pop_size, pop_max, pop_min, V_max, V_min)      # 种群初始化
	opt = Optimizer(max_iter, swarm, fitness_function)                      # 优化器初始化
	if weight:
		opt.fit(weight)      # 优化迭代
	else:
		opt.fit()
	
	# plt.figure(figsize=(5, 3))          # 结果可视化
	# plt.loglog(range(max_iter), opt.yy)
	# plt.title('PSO')
	# plt.ylim([1e-300, 10])
	# plt.grid()
	return opt.yy


# @Timer
def quantum_particle_swarm_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None):
	from swarm.particle import QuantumParticleSwarm
	
	swarm = QuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, fitness_function)
	if alpha:
		opt.fit(alpha)
	else:
		opt.fit()
	
	# plt.figure(figsize=(5, 3))
	# plt.loglog(range(max_iter), opt.yy)
	# plt.title('QPSO')
	# plt.ylim([1e-300, 10])
	# plt.grid()
	return opt.yy


# @Timer
def revised_quantum_particle_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None, beta=None):
	from swarm.particle import RevisedQuantumParticleSwarm
	
	swarm = RevisedQuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, evaluator=fitness_function)
	if alpha and beta:
		opt.fit(alpha, beta)
	else:
		opt.fit()

	# plt.figure(figsize=(5, 3))
	# plt.loglog(range(max_iter), opt.yy)
	# plt.title('RQPSO')
	# plt.ylim([1e-300, 10])
	# plt.grid()
	return opt.recoder.fitness_rec.doc


if __name__ == '__main__':
	# 超参数配置
	D_ = 2
	pop_size_ = 50
	max_iter_ = 2000
	pop_max_ = np.array([10, 10])
	pop_min_ = np.array([-10, -10])
	V_max_ = np.array([1, 1])
	V_min_ = np.array([-1, -1])
	evaluator = FitnessFunction2

	pso = particle_swarm_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, V_max_, V_min_, evaluator)
	qpso = quantum_particle_swarm_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)
	rqpso = revised_quantum_particle_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	xpso, ypso = [], []
	xqpso, yqpso = [], []
	xrqpso, yrqpso = [], []
	
	ln_pso, = ax.semilogy([], [], 'r--', animated=False, label='PSO')
	ln_qpso, = ax.semilogy([], [], 'g--', animated=False, label='QPSO')
	ln_rqpso, = ax.semilogy([], [], 'b--', animated=False, label='RQPSO')
	"""可视化功能封装"""
	
	def init():
		ax.set_xlim(1e-1, max_iter_)  # 设置x轴的范围pi代表3.14...圆周率，
		ax.set_ylim(1e-310, 10e2)  # 设置y轴的范围
		return ln_pso, ln_qpso, ln_rqpso    # 返回曲线
	
	
	def update(n):
		# xpso.append(n)  # 将每次传过来的n追加到xdata中
		# xqpso.append(n)
		# xrqpso.append(n)
		# ypso.append(pso[n])
		# yqpso.append(qpso[n])
		# yrqpso.append(rqpso[n])
		ln_pso.set_data(range(n), pso[:n])
		ln_qpso.set_data(range(n), qpso[:n])
		ln_rqpso.set_data(range(n), rqpso[:n])
		return ln_pso, ln_qpso, ln_rqpso
	
	ani = FuncAnimation(
		fig, update, frames=max_iter_,
		init_func=init, blit=False, repeat=False, interval=0,
	)
	
	plt.legend(loc='lower right', fontsize=16)
	plt.xlabel('iteration number', fontsize=16)
	plt.ylabel('fitness value', fontsize=16)
	
	plt.tight_layout()
	plt.show()
