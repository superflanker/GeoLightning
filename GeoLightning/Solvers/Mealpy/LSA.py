"""
EELT 7019 - Applied Artificial Intelligence
===========================================

Lightning Search Algorithm (LSA)

Summary
-------
This module implements the Lightning Search Algorithm (LSA), a nature-inspired 
metaheuristic based on the behavior of electrical discharges in the atmosphere. 
It combines three movement strategies — Step Leader (SL), Space Projectile (SP), 
and Lead Projectile (LP) — to explore and exploit the solution space.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- LSA class: main optimizer logic
- Step Leader, Space Projectile, and Lead Projectile sampling mechanisms

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- random
- mealpy
"""


from mealpy import Optimizer
import numpy as np
import random


class LSA(Optimizer):

    """
    Lightning Search Algorithm (LSA)

    A metaheuristic optimization algorithm inspired by lightning dynamics, 
    combining uniform (step leader), exponential (space projectile), and 
    Gaussian (lead projectile) perturbations to balance exploration and exploitation.

    Parameters
    ----------
    problem : object
        Optimization problem instance with defined objective function and bounds.
    epoch : int, optional
        Maximum number of iterations (generations). Default is 1000.
    pop_size : int, optional
        Number of individuals in the population. Default is 50.
    **kwargs : dict, optional
        Additional keyword arguments passed to the base Optimizer class.

    Attributes
    ----------
    g_best : Agent
        Best solution found during the optimization process.
    """

    def __init__(self, problem, epoch=1000, pop_size=50, **kwargs):
        super().__init__(problem, epoch, pop_size, **kwargs)

    def evolve(self, pop=None):
        for agent in pop:
            current_pos = agent.solution

            # Atualiza g_best em cada agente
            g_best = self.g_best.solution

            # Step leader (SL) usando uniforme
            sl_new = np.random.uniform(
                self.problem.lb, self.problem.ub, self.problem.n_dims)

            # Space projectile (SP) usando exponencial
            mu = np.linalg.norm(g_best - current_pos)
            sp_offset = np.random.exponential(mu, self.problem.n_dims)
            direction = np.random.choice([-1, 1], self.problem.n_dims)
            sp_new = current_pos + direction * sp_offset

            # Lead projectile (LP) usando gaussiana
            sigma = mu / 2
            lp_offset = np.random.normal(
                loc=0, scale=sigma, size=self.problem.n_dims)
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
