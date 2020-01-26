"""
搜索过程可视化
"""

from evaluator.base import FitnessFunction2
from .recorder import Recorder
from abc import ABCMeta, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as mg
from matplotlib.animation import FuncAnimation
# plt.style.use('ggplot')

global bound


class BaseVisual(metaclass=ABCMeta):
	"""动态可视化模板类"""

	@abstractmethod
	def __init__(self, *args):
		pass

	@abstractmethod
	def _plot_canvas(self, *args):
		pass

	@abstractmethod
	def _init_func(self, *args):
		pass

	@abstractmethod
	def _update(self, *args):
		pass

	@abstractmethod
	def _visual(self, *args):
		pass


class GbestVisual(BaseVisual):
	"""批量全局最优收敛过程可视化"""
	
	def __init__(self, loc_recoder: list, fit_recoder: list):
		super(GbestVisual, self).__init__()
		assert len(loc_recoder) == len(fit_recoder) == 3  # 可视化排版限制，仅限同时画3个算法的收敛过程
		self._loc_recoder = loc_recoder
		self._fit_recoder = fit_recoder
		self.epoch = max([item.epoch for item in self._fit_recoder])  # 取最大的epoch作为可视化边界
		self.D = [item.D for item in self._loc_recoder]  # 通常维度是相同的……
		
		self._plot_canvas()
		self._visual()
	
	def _plot_canvas(self):
		self._fig = plt.figure(figsize=(12, 6))
		plt.subplots_adjust(top=0.969, bottom=0.065, left=0.059, right=0.973, hspace=0.278, wspace=0.423)
		self._gs = mg.GridSpec(3, 4)  # 3行4列的栅格
		
		# 左子图：所有算法的适应度变化
		self.fit_ax = self._fig.add_subplot(self._gs[:, :2])  # 全局最优适应度值收敛过程
		self.fit_ln = [
			self.fit_ax.semilogy(
				[], [], linestyle='--', animated=False, alpha=0.9, label=alg_rec.name
			)[0] for alg_rec in self._fit_recoder
		]
		plt.legend(loc='lower left')
		
		# 右子图1：算法1的全局最优个体位置在不同维度上的变化
		self.alg1_ax = self._fig.add_subplot(self._gs[0, 2:])
		self.alg1_ln = [
			self.alg1_ax.semilogy(
				[], [], linestyle=' ', marker='.', animated=False,
				label='dim {}'.format(i + 1), alpha=0.5,
			)[0] for i in range(self.D[0])
		]
		plt.legend()
		
		# 右子图2：算法2的全局最优个体位置在不同维度上的变化
		self.alg2_ax = self._fig.add_subplot(self._gs[1, 2:])
		self.alg2_ln = [
			self.alg2_ax.semilogy(
				[], [], linestyle=' ', marker='.', animated=False,
				label='dim {}'.format(i + 1), alpha=0.5,
			)[0] for i in range(self.D[1])
		]
		plt.legend()
		
		# 右子图3：算法3的全局最优个体位置在不同维度上的变化
		self.alg3_ax = self._fig.add_subplot(self._gs[2, 2:])
		self.alg3_ln = [
			self.alg3_ax.semilogy(
				[], [], linestyle=' ', marker='.', animated=False,
				label='dim {}'.format(i + 1), alpha=0.5,
			)[0] for i in range(self.D[2])
		]
		plt.legend()
		
		plt.subplots_adjust(
			top=0.936,
			bottom=0.065,
			left=0.059,
			right=0.973,
			hspace=0.291,
			wspace=0.423
		)
		
		return self.fit_ln + self.alg1_ln + self.alg2_ln + self.alg3_ln  # 曲线/描点列表组合
	
	def _init_func(self):
		self.fit_ax.set_xlim(0, self.epoch)
		self.fit_ax.set_ylim(1e-310, 1e2)
		self.alg1_ax.set_xlim(0, self.epoch)
		self.alg1_ax.set_ylim(1e-310, 1e2)
		self.alg2_ax.set_xlim(0, self.epoch)
		self.alg2_ax.set_ylim(1e-310, 1e2)
		self.alg3_ax.set_xlim(0, self.epoch)
		self.alg3_ax.set_ylim(1e-310, 1e2)
		return self.fit_ln + self.alg1_ln + self.alg2_ln + self.alg3_ln
	
	def _update(self, epc):
		# 算法适应度值曲线更新
		
		for i in range(3):
			self.fit_ax.set_title('Times: {}'.format(epc))
			self.fit_ln[i].set_data(range(epc), self._fit_recoder[i].doc[:epc])
		
		# 不同算法在各自维度曲线更新
		for i, alg_ln in enumerate([self.alg1_ln, self.alg2_ln, self.alg3_ln]):
			for j in range(self.D[i]):
				alg_ln[j].set_data(range(epc), self._loc_recoder[i].doc[:epc, j])
		
		return self.fit_ln + self.alg1_ln + self.alg2_ln + self.alg3_ln
	
	def _visual(self):
		_ani = FuncAnimation(
			self._fig, self._update, frames=self.epoch,
			init_func=self._init_func, blit=False, repeat=False, interval=10
		)
		# plt.tight_layout()
		# _ani.save('gbest.gif', fps=30)
		plt.show()


class PbestVisual(BaseVisual):
	"""种群搜索过程可视化，仅在维度为 2 时可用"""
	
	def __init__(self, pbest_recoder: list):
		super(PbestVisual, self).__init__()
		assert len(pbest_recoder) == 3  # 限制不超过3个算法
		for rec in pbest_recoder:
			assert rec.doc.shape[-1] == 2  # 维度限制为 2
		self._pbest_recoder = pbest_recoder
		self.epoch = max([item.epoch for item in self._pbest_recoder])  # 取最大的epoch作为可视化边界
		self.D = 2
		
		self._plot_canvas()
		self._visual()
	
	def _plot_canvas(self):
		self._fig = plt.figure(figsize=(14, 4.5))
		
		self.alg1_ax = self._fig.add_subplot(131)
		self.alg1_ln, = self.alg1_ax.plot([], [], linestyle=' ', marker='.', animated=False, alpha=0.6)
		self.alg2_ax = self._fig.add_subplot(132)
		self.alg2_ln, = self.alg2_ax.plot([], [], linestyle=' ', marker='.', animated=False, alpha=0.6)
		self.alg3_ax = self._fig.add_subplot(133)
		self.alg3_ln, = self.alg3_ax.plot([], [], linestyle=' ', marker='.', animated=False, alpha=0.6)
		
		plt.subplots_adjust(
			top=0.914,
			bottom=0.087,
			left=0.048,
			right=0.978,
			hspace=0.2,
			wspace=0.217
		)
		# plt.tight_layout()
		return self.alg1_ln, self.alg2_ln, self.alg3_ln
	
	def _init_func(self):
		self.alg1_ax.set_xlim(-10, 10)
		self.alg1_ax.set_ylim(-10, 10)
		self.alg1_ax.plot([-10, 10], [0, 0], linestyle='--', color='black', alpha=0.4)
		self.alg1_ax.plot([0, 0], [-10, 10], linestyle='--', color='black', alpha=0.4)
		self.alg2_ax.set_xlim(-10, 10)
		self.alg2_ax.set_ylim(-10, 10)
		self.alg2_ax.plot([-10, 10], [0, 0], linestyle='--', color='black', alpha=0.4)
		self.alg2_ax.plot([0, 0], [-10, 10], linestyle='--', color='black', alpha=0.4)
		self.alg3_ax.set_xlim(-10, 10)
		self.alg3_ax.set_ylim(-10, 10)
		self.alg3_ax.plot([-10, 10], [0, 0], linestyle='--', color='black', alpha=0.4)
		self.alg3_ax.plot([0, 0], [-10, 10], linestyle='--', color='black', alpha=0.4)
		return self.alg1_ln, self.alg2_ln, self.alg3_ln
	
	def _update(self, epc):
		if epc < 100:
			self.alg1_ax.set_xlim(-1, 1)
			self.alg1_ax.set_ylim(-1, 1)
			self.alg2_ax.set_xlim(-1, 1)
			self.alg2_ax.set_ylim(-1, 1)
			self.alg3_ax.set_xlim(-1, 1)
			self.alg3_ax.set_ylim(-1, 1)
		elif epc < 500:
			self.alg1_ax.set_xlim(-1e-1, 1e-1)
			self.alg1_ax.set_ylim(-1e-1, 1e-1)
			self.alg2_ax.set_xlim(-1e-5, 1e-5)
			self.alg2_ax.set_ylim(-1e-5, 1e-5)
			self.alg3_ax.set_xlim(-1e-50, 1e-50)
			self.alg3_ax.set_ylim(-1e-50, 1e-50)
		elif epc < 1000:
			self.alg1_ax.set_xlim(-1E-2, 1E-2)
			self.alg1_ax.set_ylim(-1E-2, 1E-2)
			self.alg2_ax.set_xlim(-1E-20, 1E-20)
			self.alg2_ax.set_ylim(-1e-20, 1e-20)
			self.alg3_ax.set_xlim(-1e-100, 1e-100)
			self.alg3_ax.set_ylim(-1e-100, 1e-100)
		else:
			self.alg1_ax.set_xlim(-1e-2, 1e-2)
			self.alg1_ax.set_ylim(-1e-2, 1e-2)
			self.alg2_ax.set_xlim(-1e-60, 1e-60)
			self.alg2_ax.set_ylim(-1e-60, 1e-60)
			self.alg3_ax.set_xlim(-1e-150, 1e-150)
			self.alg3_ax.set_ylim(-1e-150, 1e-150)
		
		self.alg1_ax.set_title('{} {}'.format(self._pbest_recoder[0].name, epc))
		self.alg2_ax.set_title('{} {}'.format(self._pbest_recoder[1].name, epc))
		self.alg3_ax.set_title('{} {}'.format(self._pbest_recoder[2].name, epc))
		
		self.alg1_ln.set_data(self._pbest_recoder[0].doc[epc, :, 0], self._pbest_recoder[0].doc[epc, :, 1])
		self.alg2_ln.set_data(self._pbest_recoder[1].doc[epc, :, 0], self._pbest_recoder[1].doc[epc, :, 1])
		self.alg3_ln.set_data(self._pbest_recoder[2].doc[epc, :, 0], self._pbest_recoder[2].doc[epc, :, 1])
		return self.alg1_ln, self.alg2_ln, self.alg3_ln
	
	def _visual(self):
		_ani = FuncAnimation(
			self._fig, self._update, frames=self.epoch,
			init_func=self._init_func, blit=False, repeat=False, interval=10
		)
		# plt.tight_layout()
		# _ani.save("pbest.gif", fps=30)
		plt.show()


class AlgorithmVisual(BaseVisual):
	"""
	单算法优化过程可视化
	包括适应度值变化，每个维度的搜索变化，每个个体的坐标变化(仅限2D情况)
	"""

	def __init__(self, recorder: Recorder, fun: FitnessFunction2):
		super(AlgorithmVisual, self).__init__()
		self._recorder = recorder
		self._fun = fun     # 适应度函数接口
		self._plot_canvas()
		self._visual()

	@property
	def epoch(self): return self._recorder.epoch

	@property
	def step(self): return self._recorder.rec_step

	@property
	def N(self): return self._recorder.N

	@property
	def D(self): return self._recorder.D

	@property
	def t(self): return self._recorder.t

	@property
	def fitness(self): return self._recorder.fitness_rec.doc

	@property
	def gbest(self): return self._recorder.gbest_rec.doc

	@property
	def pbest(self): return self._recorder.pbest_rec.doc

	def _plot_canvas(self):
		self._fig = plt.figure(figsize=(16, 8))
		if self.D == 2:     # 2-D 情况可以画粒子的位置变化
			self._gs = mg.GridSpec(2, 2)

			# 子图1：适应度值变化
			self.fit_ax = self._fig.add_subplot(self._gs[0, 1])  # 全局最优适应度值收敛过程
			self.fit_ln, = self.fit_ax.semilogy(
				[], [], linestyle='--', animated=False, alpha=0.9,
			)

			# 子图2：每个维度的收敛过程
			self.dim_ax = self._fig.add_subplot(self._gs[1, 1])
			self.dim_ln = [
				self.dim_ax.plot(
					[], [], linestyle='-', marker='.', animated=False,
					label='dim {}'.format(i + 1), alpha=0.2
				)[0] for i in range(self.D)
			]
			plt.legend()

			# 子图3：种群收敛过程可视化
			self.swarm_ax = self._fig.add_subplot(self._gs[:, 0])
			self.swarm_ln, = self.swarm_ax.plot(
				[], [], linestyle=' ', marker='.', color='black', animated=False,
			)

			return self.fit_ln, self.dim_ln, self.swarm_ln

		else:       # 非 2-D 情况仅可视化适应度值和每个维度的变化
			# 子图1：适应度值变化
			self.fit_ax = self._fig.add_subplot(121)
			self.fit_ln, = self.fit_ax.semilogy(
				[], [], linestyle='-', animated=False, alpha=0.9,
			)

			# 子图2：每个维度的收敛过程
			self.dim_ax = self._fig.add_subplot(122)
			self.dim_ln = [
				self.dim_ax.plot(
					[], [], linestyle='--', marker='.', animated=False,
					label='dim {}'.format(i + 1), alpha=0.4,
				)[0] for i in range(self.D)
			]
			plt.legend()

			return self.fit_ln, self.dim_ln,

	def _init_func(self):

		plt.subplots_adjust(
			top=0.952,
			bottom=0.079,
			left=0.057,
			right=0.98,
			hspace=0.22,
			wspace=0.162
		)
		self.fit_ax.set_ylabel("Fitness", fontsize=14)
		self.fit_ax.set_xlabel("Iteration", fontsize=14)
		self.fit_ax.set_xlim(0, self.epoch * self.step)
		self.fit_ax.set_ylim(1e-300, 1e2)
		self.fit_ax.set_title(self._recorder.name, fontsize=14)
		self.fit_ax.grid(True)

		self.dim_ax.set_ylabel("Global best", fontsize=14)
		self.dim_ax.set_xlabel("Iteration", fontsize=14)
		self.dim_ax.set_xlim(0, self.epoch * self.step)
		self.dim_ax.set_ylim(-1e1, 1e1)
		self.dim_ax.grid(True)

		if self.D == 2:
			# 边界与标识线
			self.swarm_ax.set_title("Particle swarm position", fontsize=14)
			self.swarm_ax.set_xlabel("x1", fontsize=14)
			self.swarm_ax.set_ylabel("x2", fontsize=14)
			self.swarm_ax.set_xlim(-1, 1)
			self.swarm_ax.set_ylim(-1, 1)
			# self.swarm_ax.grid(True)

			self.swarm_ax.plot([-1, 1], [0, 0], linestyle='--', color='black', alpha=0.8)
			self.swarm_ax.plot([0, 0], [-1, 1], linestyle='--', color='black', alpha=0.8)

			return self.fit_ln, self.dim_ln, self.swarm_ln
		else:
			return self.fit_ln, self.dim_ln

	def _update(self, epc):
		self.fit_ln.set_data(self.t[:epc], self.fitness[:epc])
		for j, ln in enumerate(self.dim_ln):
			ln.set_data(self.t[:epc], self.gbest[:epc, j])

		if self.D == 2:

			# 等高线
			self.swarm_ax.cla()

			global bound
			bound = np.max([np.abs(self.pbest[epc, :, :]).max() * 1.2, 1e-30]) if epc % 5 == 0 else bound

			self.swarm_ax.plot([-bound, bound], [0, 0], linestyle='--', color='grey', alpha=0.8)
			self.swarm_ax.plot([0, 0], [-bound, bound], linestyle='--', color='grey', alpha=0.8)
			self.swarm_ax.plot(
				self.pbest[epc, :, 0], self.pbest[epc, :, 1],
				linestyle=' ', marker='.', color='black', animated=False
			)

			x, y = np.meshgrid(np.linspace(-bound, bound, 50), np.linspace(-bound, bound, 50))
			z = self._fun.infer(np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)).reshape(50, 50)
			c = plt.contour(x, y, z, 8, alpha=0.6)
			plt.clabel(c, inline=True, fontsize=12)
			self.swarm_ax.set_xlim(-bound, bound)
			self.swarm_ax.set_ylim(-bound, bound)
			plt.xticks([-bound, 0, bound])
			plt.yticks([-bound, 0, bound])



			plt.axis('equal')

			return self.fit_ln, self.dim_ln, self.swarm_ln
		else:
			return self.fit_ln, self.dim_ln

	def _visual(self):
		_ani = FuncAnimation(
			self._fig, self._update, frames=self.epoch,
			init_func=self._init_func, blit=False, repeat=True, interval=100
		)
		plt.tight_layout()
		# _ani.save("pbest.gif", fps=30)
		plt.show()
