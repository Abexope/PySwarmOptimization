# PySwarmOptimization

基于Python3开发的群体智能优化算法库

---

## 1 粒子群优化

粒子群优化(Particle Swarm Optimization，PSO)的思想是模拟鸟类集群在大自然中协同觅食的过程。在 $N$ 维搜索空间中抽象出若干抽象实体，称为“粒子”。所有的粒子都有一个由目标函数 $f()$ 决定的适应度值(fitness value)，即目标函数值。每个粒子在空间中所处的位置被称为目标函数的一个“候选解”。假设粒子群包含 $M$ 个粒子，由这些粒子组成一个粒子群集合
$$
X = \{ X_1, X_2, ..., X_i ,... , X_M \}
$$
其中 $i$ 满足 $1 \leqslant i \leqslant M$。在第 $t$ 时刻，第 $i$ 个粒子在 $N$ 为空间中的位置向量表示为
$$
X_i(t) = [X_{i,1}(t),X_{i,2}(t),...,X_{i,j}(t),...,X_{i,N}(t)]
$$
其中 $j$ 满足 $1 \leqslant j \leqslant N$。每个粒子都根据自身和群体的搜索经历进行调整，逐代优化。在每轮迭代中，粒子会追随两个最好的位置，一是粒子本身迄今找到的最好位置，成为个体最好(Personal Best Position，$pbest$)位置；二是整个粒子群迄今为止找到的最好位置，称为全局最好位置(Global Best Position，$gbest$)。第 $i$ 个粒子的个体最好位置表示为
$$
P_i(t) = [P_{i,1}(t),P_{i,2}(t),...,P_{i,j}(t),...,P_{i,N}(t)]
$$
全局最好位置表示为
$$
G(t) = [G_1(t),G_2(t),...,G_j(t),...,G_N(t)]
$$
令 $g$ 为处于全局最好位置粒子的下标，$g \in \{ 1,2,..M \}$，即 $G(t) = P_g(t)$。个体最好位置 $pbest$ 与全局最好位置 $gbest$ 的选取方式取决于粒子所处位置的目标函数值 $f(X_i(t))$。以最小化问题为例
$$
\min f(x)
$$
粒子 $i$ 的 $pbest$ 更新方式为
$$
P_i(t)= \begin{cases}
X_i(t),& f(X_i(t))<f(P_i(t-1))\\
P_i(t-1),& f(X_i(t)) \geqslant f(P_i(t-1))
\end{cases}
$$
$gbest$ 的更新方式为
$$
g=\arg \min f(P_i(t)), 1 \leqslant i \leqslant M  \\
G(t)=P_g(t)
$$
以上是粒子群优化及其衍生优化算法所必须包含的基本属性。后续将根据不同算法阐述各自的种群进化过程(方程)。

### 1.1 经典行为进化

经典PSO算法最初由Eberhart和Kennedy提出，在基本属性的基础上，粒子增加了“速度”属性作为。对于第 $i$ 个粒子，速度向量为
$$
V_i(t) = [V_{i,1}(t),V_{i,2}(t),...,V_{i,j}(t),...,V_{i,N}(t)]
$$
对于第 $i$ 个粒子中的第 $j$ 个维度，PSO算法的进化方程
$$
V_{i,j}(t+1)=V_{i,j}(t)+c_1·r_{1,j}(t)·[P_{i,j}(t)-X_{i,j}(t)]+c_2·r_{2,j}(t)·[G_j(t)-X_{i,j}(t)] \\
X_{i,j}(t+1) = X_{i,j}(t)+V_{i,j}(t+1)
$$
其中，$c_1$ 和 $c_2$ 为加速系数，$r_{1,j}$ 和 $r_{2,j}$ 为 $[0, 1]$ 区间内服从均匀分布的随机数。

### 1.2 量子行为进化

在量子力学中，使用粒子的束缚态来刻画粒子群的聚集性/收敛性。产生束缚态的原因是在粒子运动的中心存在某种吸引势场。为此可以建立一个量子化的吸引势场，从而束缚粒子个体以使群体具呈现出聚集性。刻画粒子的收敛过程是以 $pbest$ 和 $gbest$ 做随机加权得到的位置向量作为吸引点
$$
p_{i,j}(t)=\phi_{i,j}(t)·P_{i,j}(t)+(1-\phi_{i,j}(t)·G_j(t)) \\
\phi_{i,j}(t) \sim \text{U}(0, 1)
$$
粒子的进化表现为受到粒子的吸引，进化方程为
$$
X_{i,j}(t+1)=p_{i,j}(t) \pm \alpha·\lvert C_j(t) - X_{i,j}(t) \rvert ·\ln[\frac{1}{u_{i,j}(t)}]
$$
其中
$$
C_{i,j}(t) = \frac{1}{M}\sum_{i=1}^{M}{P_{i,j}(t)}
$$
表示所有 $P_i,j(t)$ 的平均。采用量子行为进化方程的PSO算法称为量子行为粒子群优化(Quantum-behaved Particle Swarm Optimization，QPSO)算法。

### 1.3  量子行为拓展

QPSO算法的进化方程可以归纳为
$$
X_{i,j}(t+1)=p_{i,j}(t) \pm \alpha·\lvert C_j(t) - X_{i,j}(t) \rvert ·\ln[\frac{1}{u_{i,j}(t)}] \\
=p_{i,j}(t)+A_{i,j}(t)
$$
其中
$$
A_{i,j}(t)=\pm \alpha·\lvert C_j(t) - X_{i,j}(t) \rvert ·\ln[\frac{1}{u_{i,j}(t)}]
$$
相当于一个服从拉普拉斯分布的随机变量。所以，一种提升QPSO算法收敛性能的思想是提升算法在进化过程中的不确定性。令
$$
A_{i,j}(t)=a_{i,j}(t)+b_{i,j}(t)
$$
其中
$$
a_{i,j}(t)=\pm \alpha·\lvert C_j(t) - X_{i,j}(t) \rvert ·\ln[\frac{1}{u_{i,j}(t)}] \\
b_{i,j}(t)=\beta·\lvert C_{i,j}(t)-X_{i,j}(t) \rvert · n(t) \\
n(t) \sim \text N(0,1)
$$
进化方程改进为
$$
X_{i,j}(t+1)=p_{i,j}(t) \pm \alpha·\lvert C_j(t) - X_{i,j}(t) \rvert ·\ln[\frac{1}{u_{i,j}(t)}]+\beta·\lvert C_{i,j}(t)-X_{i,j}(t) \rvert · n(t)
$$
采用此种进化方程的算法被称之为具有混合概率分布的量子行为粒子群优化(Revised Quantum-behaved Particle Swarm Optimization, RQPSO)算法。

## 2 代码架构

### 2.1 evaluator

模板类



### 2.2 swarm

模板类



### 2.3 optimizer



### 2.4 support

#### 2.4.1 recoder.py

模板类



#### 2.4.2 visualize.py

模板类







## 3. 测试样例













目前已经开发的算法包括：

- 粒子群算法(Particle Swarm Optimization，PSO)
- 量子行为粒子群算法(Quantum-behaved Particle Swarm Optimization，QPSO)
- 混合量子行为粒子群算法(Revised Quantum-behaved Particle Swarm Optimization，RQPSO)

一些可视化结果：

- 优化目标函数：

$$
\min f(\mathbf{x})=\mathbf{x^Tx}
$$

- 全局最优收敛过程示意

![](.\doc\gbest.png)

- 种群搜索过程示意

![](.\doc\pbest.png)

- PSO:

![](.\doc\pso.gif)