"""
Mealpy FHO Wrapper
==================

Fire Hawk Optimizer (FHO)

Summary
-------
This module implements the Fire Hawk Optimizer (FHO), a population-based metaheuristic
inspired by the natural behavior of fire hawks and their hunting strategy. It divides
the population into two classes — fire hawks and prey — and updates their positions 
based on predator-prey dynamics and safe regions.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- FHO class: optimization algorithm implementation
- Fire hawk and prey update mechanisms
- Dynamic exploitation of safe zones

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


from mealpy import Optimizer
from mealpy.utils.agent import Agent
import numpy as np


class FHO(Optimizer):
    """
    Fire Hawk Optimizer (FHO)

    A nature-inspired metaheuristic algorithm based on the hunting and fire-spreading
    behavior of fire hawks. This optimizer models interactions between fire hawks and
    prey, including dynamic territory formation, safe zone calculation, and adaptive
    movement strategies.

    Parameters
    ----------
    epoch : int, optional
        Maximum number of iterations (generations). Default is 1000.
    pop_size : int, optional
        Total number of agents (hawks + prey). Default is 50.
    **kwargs : dict, optional
        Additional keyword arguments passed to the base Optimizer class.

    Attributes
    ----------
    n_hawks : int
        Number of fire hawks in the population (30% of pop_size).
    n_preys : int
        Number of prey in the population (70% of pop_size).
    """

    def __init__(self, epoch=1000, pop_size=50, **kwargs):
        super().__init__(**kwargs)
        self.epoch = self.validator.check_int("epoch", epoch, [1, 100000])
        self.pop_size = self.validator.check_int("pop_size", pop_size, [5, 10000])
        self.n_hawks = int(0.3 * self.pop_size)
        self.n_preys = self.pop_size - self.n_hawks
        self.set_parameters(["epoch", "pop_size"])
        self.sort_flag = False

    def initialize_variables(self):
        self.hawks = []
        self.preys = []
        hawks = [np.random.uniform(self.problem.lb, self.problem.ub) for _ in range(self.n_hawks)]
        preys = [np.random.uniform(self.problem.lb, self.problem.ub) for _ in range(self.n_preys)]
        for _, new_pos in enumerate(hawks):
            target = self.problem.get_target(new_pos)
            new_agent = Agent(new_pos, target)
            self.hawks.append(new_agent)
        for _, new_pos in enumerate(preys):
            target = self.problem.get_target(new_pos)
            new_agent = Agent(new_pos, target)
            self.preys.append(new_agent)

    def compute_distance(self, x1, x2):
        """
        Compute the Euclidean distance between two solution vectors.

        This distance metric is used to associate each prey with the closest
        fire hawk based on spatial proximity in the solution space.

        Parameters
        ----------
        x1 : ndarray
            First solution vector.
        x2 : ndarray
            Second solution vector.

        Returns
        -------
        float
            Euclidean distance between x1 and x2.
        """
        return np.linalg.norm(x1 - x2)

    def update_hawks(self):
        """
        Update the positions of all fire hawks.

        Each hawk moves toward the global best solution while being repelled by a randomly
        selected hawk, encouraging both convergence and diversity in the population.
        Position updates are subject to boundary constraints and fitness evaluation.
        """
        for i, hawk in enumerate(self.hawks):
            r1, r2 = np.random.rand(2)
            random_hawk = self.hawks[np.random.randint(self.n_hawks)]
            new_pos = hawk.solution + r1 * (self.g_best.solution - hawk.solution) - r2 * random_hawk.solution
            new_pos = self.amend_solution(new_pos)
            target = self.problem.get_target(new_pos)
            new_agent = Agent(new_pos, target)
            if self.compare_target(new_agent.target, hawk.target):
                self.hawks[i] = new_agent

    def update_hawks_territories(self, territories):
        """
        Assign each prey to the territory of the closest fire hawk.

        The Euclidean distance is used to associate preys with the nearest hawk,
        forming dynamic territorial clusters that reflect local exploitation zones.

        Parameters
        ----------
        territories : list of list
            A list where each sublist contains the preys assigned to a hawk.

        Returns
        -------
        list of list
            Updated territory assignment of preys per hawk.
        """
        for prey in self.preys:
            distances = [self.compute_distance(prey.solution, hawk.solution) for hawk in self.hawks]
            idx = int(np.argmin(distances))
            territories[idx].append(prey)
        return territories
    
    def update_preys_inside_territory(self, 
                                      new_preys, 
                                      territories):
        """
        Update the positions of preys located within a hawk's territory.

        Each prey is attracted to its assigned hawk and repelled from the mean position
        (safe place) of all preys in the same territory, balancing convergence and dispersion.

        Parameters
        ----------
        new_preys : list
            A list to accumulate the updated prey agents.
        territories : list of list
            Territory assignment for each hawk.
        """
        for i, hawk in enumerate(self.hawks):
            if not territories[i]:
                continue
            safe_place = np.mean([p.solution for p in territories[i]], axis=0)
            for prey in territories[i]:
                r3, r4 = np.random.rand(2)
                new_pos = prey.solution + r3 * hawk.solution - r4 * safe_place
                new_pos = self.amend_solution(new_pos)
                target = self.problem.get_target(new_pos)
                new_agent = Agent(new_pos, target)
                if self.compare_target(new_agent.target, prey.target):
                    new_preys.append(new_agent)
                else:
                    new_preys.append(prey)

    def update_preys_outside_territory(self, new_preys, territories):
        """
        Update the positions of preys not assigned to any hawk territory (escaped preys).

        Each escaped prey moves toward a randomly selected hawk and away from the global
        safe zone, computed as the mean of all escaped prey positions. This enables exploration
        in globally less-exploited areas.

        Parameters
        ----------
        new_preys : list
            A list to accumulate the updated prey agents.
        territories : list of list
            Territory assignment for each hawk (used to identify escaped preys).
        """
        all_preys = [p for group in territories for p in group]
        escaped_preys = [p for p in self.preys if p not in all_preys]
        if escaped_preys:
            global_safe_place = np.mean([p.solution for p in escaped_preys], axis=0)
            for prey in escaped_preys:
                hawk = self.hawks[np.random.randint(self.n_hawks)]
                r5, r6 = np.random.rand(2)
                new_pos = prey.solution + r5 * hawk.solution - r6 * global_safe_place
                new_pos = self.amend_solution(new_pos)
                target = self.problem.get_target(new_pos)
                new_agent = Agent(new_pos, target)
                if self.compare_target(new_agent.target, prey.target):
                    new_preys.append(new_agent)
                else:
                    new_preys.append(prey)
    

    def evolve(self, epoch):
        """
        Perform one iteration of the Fire Hawk Optimizer.

        This method updates the positions of both hawks and preys based on dynamic
        rules of fire hawk hunting. It consists of:

        1. Updating hawk positions via attraction to the global best and repulsion from other hawks;
        2. Assigning each prey to the nearest hawk's territory based on Euclidean distance;
        3. Updating the position of preys within territories by moving them toward their assigned hawk
           and away from a computed safe zone (local mean);
        4. Updating the position of escaped preys outside any territory by attracting them to a random hawk
           and repelling them from the global safe zone (mean of escaped preys);
        5. Updating the global best solution.

        Parameters
        ----------
        epoch : int
            Current iteration number.
        """
        self.epoch_current = epoch

        self.update_hawks()

        territories = [[] for _ in range(self.n_hawks)]
        self.update_hawks_territories(territories)

        new_preys = []

        self.update_preys_inside_territory(new_preys, territories)

        self.update_preys_outside_territory(new_preys, territories)

        self.preys = new_preys
        self.pop = self.hawks + self.preys
        _, self.g_best = self.update_global_best_agent(self.pop)
