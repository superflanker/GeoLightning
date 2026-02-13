"""
Mealpy LSA Wrapper
==================

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
- Transition Projectile, Space Projectile, and Lead Projectile sampling mechanisms

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
from mealpy.utils.agent import Agent
import numpy as np
import random


class LSA(Optimizer):

    """
    Lightning Search Algorithm (LSA)

    A metaheuristic optimization algorithm inspired by lightning dynamics,
    combining three distinct movement mechanisms to explore and exploit the
    search space: transition projectile (uniform), space projectile (exponential),
    and lead projectile (Gaussian).

    Parameters
    ----------
    problem : object
        An instance of the optimization problem, defining bounds and objective.
    epoch : int, optional
        Maximum number of iterations to execute. Default is 1000.
    pop_size : int, optional
        Number of agents (solutions) in the population. Default is 50.
    **kwargs : dict, optional
        Additional arguments passed to the base Optimizer class.

    Attributes
    ----------
    g_best : Agent
        The best solution found during the optimization.
    is_parallelizable : bool
        Indicates whether the algorithm supports parallel evaluation. Always False.
    sort_flag : bool
        Sorting flag for population ranking. Not used in this algorithm.
    """

    def __init__(self, epoch=1000, pop_size=50, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int(
            "pop_size", pop_size, [5, 10000])
        self.set_parameters(["epoch", "pop_size"])
        self.is_parallelizable = False
        self.sort_flag = False
        self.no_improve_counter = [0] * self.pop_size
        self.diversity_threshold = kwargs.get("diversity_threshold", 1e-5)
        self.diversity_fraction = kwargs.get("diversity_fraction", 0.5)

    def initialize_variables(self):
        pop = [np.random.uniform(self.problem.lb, self.problem.ub)
               for _ in range(self.pop_size)]
        self.pop = []
        for _, new_pos in enumerate(pop):
            target = self.problem.get_target(new_pos)
            new_agent = Agent(new_pos, target)
            self.pop.append(new_agent)

    def compute_mu(self, agent):
        """
        Compute the distance (mu) from a given agent to the global best.

        This value controls the magnitude of the exponential perturbation (space projectile).

        Parameters
        ----------
        agent : Agent
            Current solution agent.

        Returns
        -------
        float
            Euclidean distance between the agent and the global best.
        """
        mu = np.linalg.norm(self.g_best.solution - agent.solution)
        return mu

    def compute_sigma(self, mu):
        """
        Compute the standard deviation (sigma) for Gaussian sampling.

        Sigma is derived from mu and controls the lead projectile's spread.

        Parameters
        ----------
        mu : float
            Distance from the current agent to the global best.

        Returns
        -------
        float
            Standard deviation used for Gaussian sampling.
        """
        return mu / 2.0

        # return mu * 0.75

    def transition_projectile(self):
        """
        Generate a transition projectile (TP) candidate.

        Samples a new position uniformly within the search bounds. Represents a random
        discharge independent of current population dynamics.

        Returns
        -------
        ndarray
            A new solution vector sampled uniformly within bounds.
        """
        tp_new = np.random.uniform(self.problem.lb,
                                   self.problem.ub,
                                   self.problem.n_dims)
        return tp_new

    def space_projectile(self, agent,  mu):
        """
        Generate a space projectile (SP) candidate.

        Samples a perturbation from an exponential distribution and applies it in a
        random direction to the current agent. Explores the space based on distance to the best.

        Parameters
        ----------
        agent : Agent
            The current agent.
        mu : float
            Characteristic length scale based on distance to the global best.

        Returns
        -------
        ndarray
            A new solution vector generated via exponential displacement.
        """
        sp_offset = np.random.exponential(mu, len(self.problem.lb))
        direction = np.random.choice([-1, 1], len(self.problem.lb))
        sp_new = agent.solution + direction * sp_offset
        return sp_new

    def lead_projectile(self, sigma):
        """
        Generate a lead projectile (LP) candidate.

        Applies Gaussian perturbation to the global best position, promoting local
        exploitation of the most promising region found so far.

        Parameters
        ----------
        sigma : float
            Standard deviation controlling the Gaussian spread.

        Returns
        -------
        ndarray
            A new solution vector generated via Gaussian perturbation of the global best.
        """
        lp_offset = np.random.normal(
            loc=0, scale=sigma, size=self.problem.n_dims)
        lp_new = self.g_best.solution + lp_offset
        return lp_new

    def opposition_position(self, pos):
        return self.problem.ub + self.problem.lb - pos

    def evolve(self, epoch=None):
        """
        Perform one iteration of the Lightning Search Algorithm (LSA).

        For each agent in the population:
        - Computes mu (distance to global best),
        - Computes sigma (Gaussian spread),
        - Generates one candidate from each projectile type (TP, SP, LP),
        - Randomly selects one,
        - Updates the agent if the new solution improves its fitness.

        Parameters
        ----------
        pop : epochs
        """

        for idx, agent in enumerate(self.pop):
            mu = self.compute_mu(agent)
            sigma = self.compute_sigma(mu)

            tp_new = self.transition_projectile()
            sp_new = self.space_projectile(agent, mu)
            lp_new = self.lead_projectile(sigma)

            # Adaptive selection: more exploitation when close to best
            choices = [tp_new, sp_new, lp_new]
            weights = [0.3, 0.4, 0.3] if mu > 10 else [0.1, 0.2, 0.7]
            pos_new = random.choices(choices, weights=weights, k=1)[0]

            pos_new = self.amend_solution(pos_new)
            target = self.problem.get_target(pos_new)
            new_agent = Agent(pos_new, target)

            if self.compare_target(new_agent.target, agent.target):
                self.pop[idx] = new_agent.copy()
