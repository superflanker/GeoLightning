"""
Stela PSO Wrapper
=================

STELA Wrapper for the Particle Swarm Optimization (PSO)

Summary
-------
This module defines the `StelaPSO` class, a specialized extension of the 
Particle Swarm Optimization (PSO) algorithm from the MEALPY library. It is 
designed to operate directly on instances of the `StelaProblem` class, which 
models the spatio-temporal localization of events detected by distributed sensors, 
such as atmospheric lightning.

Before each evolutionary iteration, the search space is adaptively refined 
based on the current best solution using the `restart_search_space()` method 
from the `StelaProblem` class.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- StelaPSO class: adaptive PSO wrapper for STELA
- evolve(): PSO iteration with dynamic bounds refinement

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- mealpy.swarm_based.PSO
- GeoLightning.Solvers.StelaProblem
"""

import numpy as np
from mealpy.swarm_based.PSO import OriginalPSO
from GeoLightning.Solvers.StelaProblem import StelaProblem

class StelaPSO(OriginalPSO):
    """
    Specialized class extending MEALPY's PSO optimizer to operate directly 
    on `StelaProblem` instances for spatio-temporal event localization.

    Before each evolutionary iteration, the search space is adaptively 
    refined based on the current best solution using the 
    `restart_search_space()` method from `StelaProblem`.

    Parameters
    ----------
    epoch : int, optional
        Number of evolutionary iterations (default is 1000).
    pop_size : int, optional
        Number of candidate solutions (default is 50).
    **kwargs : dict, optional
        Additional arguments passed to the parent class.
    """

    def evolve(self, pop=None):
        """
        Executes one iteration of the PSO algorithm with adaptive search space 
        refinement tailored to the `StelaProblem`.

        This method overrides the default MEALPY implementation, integrating 
        dynamic boundary updates based on the best solution found so far.

        Parameters
        ----------
        pop : list, optional
            Current population of agents (particles). If not provided, 
            the internal population is used.

        Raises
        ------
        TypeError
            Raised if the associated problem is not an instance of `StelaProblem`.
        """
        if not isinstance(self.problem, StelaProblem):
            raise TypeError("O problema fornecido deve ser uma instância de StelaProblem.")
        self.problem.restart_search_space()
        super().evolve(pop)
