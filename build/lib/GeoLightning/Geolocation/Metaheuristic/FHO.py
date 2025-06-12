"""
    EELT 7019 - Inteligência Artificial Aplicada
    Algoritmo Fire Hawk Optimizer (FHO)
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from mealpy import Optimizer
import numpy as np


class FHO(Optimizer):
    """
    Fire Hawk Optimizer (FHO)
    """

    def __init__(self, problem, epoch=1000, pop_size=50, **kwargs):
        super().__init__(problem, epoch, pop_size, **kwargs)

    def evolve(self, pop=None):
        X = self.get_position(pop)
        fitness = np.array([agent.fit for agent in pop])

        # Dividir em Fire Hawks (30%) e Prey (70%)
        n_hawks = max(1, int(0.3 * len(pop)))
        sorted_idx = np.argsort(fitness)
        hawks = [pop[idx] for idx in sorted_idx[:n_hawks]]
        prey = [pop[idx] for idx in sorted_idx[n_hawks:]]

        # Atualização dos Fire Hawks
        for i, hawk in enumerate(hawks):
            r1, r2 = np.random.rand(), np.random.rand()
            g_best = self.g_best.solution
            near_idx = np.random.choice([j for j in range(n_hawks) if j != i])
            hawk_near = hawks[near_idx].solution

            pos_new = hawk.solution + (r1 * (g_best - (r2 * hawk_near)))
            pos_new = self.amend_position(pos_new)
            fit_new = self.get_fitness_position(pos_new)

            if self.compare_agent(hawk.fit, fit_new):
                hawk.solution = pos_new
                hawk.fit = fit_new
                if self.compare_agent(self.g_best.fit, fit_new):
                    self.g_best = hawk.copy()

        # Safe places
        prey_positions = np.array([agent.solution for agent in prey])
        safe_place_in = np.mean(prey_positions, axis=0)
        safe_place_out = np.mean(X, axis=0)

        # Atualização dos Prey
        for agent in prey:
            hawk_idx = np.random.randint(n_hawks)

            if np.random.rand() < 0.5:
                r3, r4 = np.random.rand(), np.random.rand()
                pos_new = agent.solution + (r3 * (hawks[hawk_idx].solution - r4 * safe_place_in))
            else:
                r5, r6 = np.random.rand(), np.random.rand()
                alter_idx = np.random.choice([j for j in range(n_hawks) if j != hawk_idx])
                pos_new = agent.solution + (r5 * (hawks[alter_idx].solution - r6 * safe_place_out))

            pos_new = self.amend_position(pos_new)
            fit_new = self.get_fitness_position(pos_new)

            if self.compare_agent(agent.fit, fit_new):
                agent.solution = pos_new
                agent.fit = fit_new
                if self.compare_agent(self.g_best.fit, fit_new):
                    self.g_best = agent.copy()

        # Atualiza a população original
        pop = hawks + prey
        return pop, self.g_best
