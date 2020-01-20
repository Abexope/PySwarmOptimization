"""
搜索过程可视化
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as mg
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')


class GbestVisual:
	"""批量全局最优收敛过程可视化"""
	
	def __init__(self, loc_recoder: list, fit_recoder: list):
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


class PbestVisual:
	"""种群搜索过程可视化，仅在维度为 2 时可用"""
	
	def __init__(self, pbest_recoder: list):
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
