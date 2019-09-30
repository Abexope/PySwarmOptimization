"""
test demo
"""

from swarm.particle import Particle
from optimizer.pso import ParticleSwarmOptimization as PSO
from evaluator.base import FitnessFunction
import matplotlib.pyplot as plt


D = 2
pop_size = 50
max_iter = 100
pop_max = [10, 10]
pop_min = [-10, -10]
V_max = [1, 1]
V_min = [-1, -1]

swarm = [Particle(D, pop_max, pop_min, V_max, V_min) for _ in range(pop_size)]
pso = PSO(evaluator=FitnessFunction, swarm=swarm, population_size=pop_size)
pso.search()

plt.figure()
plt.plot(range(max_iter), pso.yy)
plt.grid()

plt.show()
