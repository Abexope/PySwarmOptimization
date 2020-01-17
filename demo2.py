"""动态可视化（功能测试版）"""
import numpy as np
from optimizer.pso import Optimizer
from evaluator.base import FitnessFunction2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')


class GbestVisual:
	"""全局最优的收敛过程可视化"""
	
	def __init__(self, loc_recoder, fit_recoder):
		self._loc_recoder = loc_recoder  # 位置记录
		self._fit_recoder = fit_recoder  # 适应度值记录
		self.epoch = loc_recoder.epoch  # 迭代次数
		self.D = loc_recoder.D  # 维度
		self._plot_canvas()  # 初始化画布
		self._visual()
	
	def _plot_canvas(self):
		self._fig = plt.figure()
		
		self.fit_ax = self._fig.add_subplot(121)  # 可视化最优适应度值收敛过程
		self.fit_ln, = self.fit_ax.semilogy([], [], linestyle='-', color='b', animated=False, label='fitness value')
		
		self.loc_ax = self._fig.add_subplot(122)  # 可视化最优位置收敛过程
		# self.loc_ln1, = self.loc_ax.semilogy([], [], linestyle=' ', marker='.', animated=False, label='D1', alpha=0.5)
		# self.loc_ln2, = self.loc_ax.semilogy([], [], linestyle=' ', marker='.', animated=False, label='D2', alpha=0.5)
		
		self.ln = tuple([
			self.loc_ax.semilogy(
				[], [], linestyle=' ', marker='.', animated=False,
				label='D{}'.format(i), alpha=0.5
			)[0] for i in range(self.D)
		])
		
		# lin_tuple = (self.fit_ln, self.loc_ln1, self.loc_ln2)
		
		return (self.fit_ln,) + self.ln
	
	def _init_func(self):
		self.fit_ax.set_xlim(0, self.epoch)
		self.fit_ax.set_ylim(1e-310, 1e2)
		self.loc_ax.set_xlim(0, self.epoch)
		self.loc_ax.set_ylim(1e-310, 1e2)
		return (self.fit_ln,) + self.ln
	
	def _update(self, epc):
		self.fit_ln.set_data(range(epc), self._fit_recoder.doc[:epc])  # 画fitness
		# self.loc_ln1.set_data(range(epc), self._loc_recoder.doc[:epc, 0])  # 画维度1
		# self.loc_ln2.set_data(range(epc), self._loc_recoder.doc[:epc, 1])  # 画维度2
		for i in range(self.D):
			self.ln[i].set_data(range(epc), self._loc_recoder.doc[:epc, i])
		return (self.fit_ln,) + self.ln
	
	def _visual(self):
		_ani = FuncAnimation(
			self._fig, self._update, frames=self.epoch,
			init_func=self._init_func, blit=False, repeat=False, interval=0,
		)
		plt.legend()
		plt.tight_layout()
		plt.show()


def revised_quantum_particle_opt(D, pop_size, max_iter, pop_max, pop_min, fitness_function, alpha=None, beta=None):
	from swarm.particle import RevisedQuantumParticleSwarm
	
	swarm = RevisedQuantumParticleSwarm(D, pop_size, pop_max, pop_min)
	opt = Optimizer(max_iter, swarm, evaluator=fitness_function)
	if alpha and beta:
		opt.fit(alpha, beta)
	else:
		opt.fit()
	
	return opt.recoder


if __name__ == '__main__':
	D_ = 2
	pop_size_ = 50
	max_iter_ = 2000
	pop_max_ = np.array([10, 10])
	pop_min_ = np.array([-10, -10])
	V_max_ = np.array([1, 1])
	V_min_ = np.array([-1, -1])
	evaluator = FitnessFunction2
	rqpso_rec = revised_quantum_particle_opt(D_, pop_size_, max_iter_, pop_max_, pop_min_, evaluator)
	
	GbestVisual(rqpso_rec.gbest_rec, rqpso_rec.fitness_rec)
	# gbv.visual()
	
	pass
