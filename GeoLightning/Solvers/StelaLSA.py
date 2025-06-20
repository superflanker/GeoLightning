"""
EELT 7019 - Applied Artificial Intelligence
===========================================

STELA Wrapper for the Lightning Search Algorithm (LSA)

Summary
-------
This module defines the class `StelaLSA`, a specialized extension of the 
Lightning Search Algorithm (LSA) optimizer from the MEALPY library. It is designed 
to directly operate on instances of the `StelaProblem` class, which models 
spatio-temporal localization problems, such as atmospheric lightning events.

Before each evolutionary iteration, the search space is adaptively refined 
based on the current best solution through the `restart_search_space()` method 
defined in `StelaProblem`.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- StelaLSA class: adaptive LSA wrapper for STELA
- evolve(): executes one iteration with dynamic bounds adjustment

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- GeoLightning.Solvers.Mealpy.LSA
- GeoLightning.Solvers.StelaProblem
"""

import numpy as np
from GeoLightning.Solvers.Mealpy.LSA import LSA
from GeoLightning.Solvers.StelaProblem import StelaProblem

class StelaLSA(LSA):
    """
    Specialized class extending MEALPY's LSA optimizer to work with 
    `StelaProblem` instances for spatio-temporal event localization.

    This wrapper integrates dynamic refinement of the search space 
    at each iteration based on the current best solution found.

    Parameters
    ----------
    problem : StelaProblem
        An instance of the STELA geolocation problem.
    epoch : int, optional
        Maximum number of iterations. Default is 1000.
    pop_size : int, optional
        Number of individuals in the population. Default is 50.
    **kwargs : dict
        Additional arguments for the base optimizer.
    """

    def evolve(self, pop=None):
        """
        Executes one iteration of the LSA algorithm with adaptive 
        search space refinement for `StelaProblem`.

        This method overrides the original `evolve()` implementation in MEALPY 
        to include dynamic bounds update via `restart_search_space()`.

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
