"""
    EELT 7019 - Inteligência Artificial Aplicada
    Algoritmo LSA - 
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from mealpy import Optimizer
import numpy as np
import random


class LSA(Optimizer):
    """
    Lightning Search Algorithm (LSA)
    """

    def __init__(self, problem, epoch=1000, pop_size=50, **kwargs):
        super().__init__(problem, epoch, pop_size, **kwargs)

    def evolve(self, pop=None):
        for agent in pop:
            current_pos = agent.solution

            # Atualiza g_best em cada agente
            g_best = self.g_best.solution

            # Step leader (SL) usando uniforme
            sl_new = np.random.uniform(self.problem.lb, self.problem.ub, self.problem.n_dims)

            # Space projectile (SP) usando exponencial
            mu = np.linalg.norm(g_best - current_pos)
            sp_offset = np.random.exponential(mu, self.problem.n_dims)
            direction = np.random.choice([-1, 1], self.problem.n_dims)
            sp_new = current_pos + direction * sp_offset

            # Lead projectile (LP) usando gaussiana
            sigma = mu / 2
            lp_offset = np.random.normal(loc=0, scale=sigma, size=self.problem.n_dims)
            lp_new = g_best + lp_offset

            # Seleciona aleatoriamente um dos três
            pos_new = random.choice([sl_new, sp_new, lp_new])

            # Limita aos bounds
            pos_new = self.amend_position(pos_new)

            # Avalia
            fit_new = self.get_fitness_position(pos_new)

            # Se melhor, aceita
            if self.compare_agent(agent.fit, fit_new):
                agent.solution = pos_new
                agent.fit = fit_new

                if self.compare_agent(self.g_best.fit, fit_new):
                    self.g_best = agent.copy()

        return pop, self.g_best
