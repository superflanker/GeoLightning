"""
EELT 7019 - Applied Artificial Intelligence
===========================================

Electrical Storm Optimization (ESO) Algorithm

Summary
-------
This module implements the Electrical Storm Optimization (ESO) algorithm, a nature-inspired 
metaheuristic that simulates the dynamic behavior of electrical storms to solve continuous 
optimization problems. It models the interactions between field intensity, resistance, 
conductivity, and ionization zones to guide the search process in the solution space.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- ESO class: core optimizer implementing the electrical storm metaphor.
- Parameters include iteration-dependent field behavior and selective ionization.

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- mealpy
"""


from mealpy import Optimizer
import numpy as np


class ESO(Optimizer):
    """
    Electrical Storm Optimization (ESO)

    A nature-inspired metaheuristic based on the physics of electrical storms,
    incorporating variables such as resistance (R), conductivity (ke), intensity (I),
    and power (P). The population is guided by the influence of the most ionized
    (promising) areas, encouraging both exploration and exploitation.

    Parameters
    ----------
    problem : object
        Optimization problem instance with defined objective function and bounds.
    epoch : int, optional
        Maximum number of iterations (generations). Default is 1000.
    pop_size : int, optional
        Number of candidate solutions (agents) in the population. Default is 50.
    **kwargs : dict, optional
        Additional keyword arguments passed to the base Optimizer class.

    Attributes
    ----------
    g_best : Agent
        Best solution found so far.
    """

    def __init__(self, problem, epoch=1000, pop_size=50, **kwargs):
        super().__init__(problem, epoch, pop_size, **kwargs)

    def evolve(self, pop=None):
        X = self.get_position(pop)

        # Field Resistance (R)
        std_dev = np.std(X, axis=0).mean()
        peak_to_peak = (np.max(X, axis=0) - np.min(X, axis=0)).mean()
        R = std_dev / peak_to_peak if peak_to_peak != 0 else 1e-6

        # Field Conductivity (ke) - Sigmoid Function
        ke = 1 / (1 + np.exp(-R))

        # Field Intensity (I) - Adaptive with iteration
        gamma = 1 / (1 + np.exp(-(1 - self.current_epoch / self.epoch)))
        I = ke * gamma

        # Storm Power (P)
        P = (R * I) / (ke + 1e-10)

        # Ionized areas - top 20%
        sorted_idx = np.argsort([agent.fit for agent in pop])
        top_k = max(1, int(len(pop) * 0.2))
        ionized = [pop[i] for i in sorted_idx[:top_k]]
        ionized_positions = np.array([agent.solution for agent in ionized])

        center_ionized = np.mean(ionized_positions, axis=0)

        for agent in pop:
            if agent in ionized:
                pos_new = agent.solution + np.random.uniform(-1, 1, self.problem.n_dims) * P
            else:
                random_step = np.random.uniform(-ke, ke, self.problem.n_dims)
                pos_new = center_ionized + random_step * P * np.exp(ke)

            pos_new = self.amend_position(pos_new)
            fit_new = self.get_fitness_position(pos_new)

            if self.compare_agent(agent.fit, fit_new):
                agent.solution = pos_new
                agent.fit = fit_new

                if self.compare_agent(self.g_best.fit, fit_new):
                    self.g_best = agent.copy()

        return pop, self.g_best
