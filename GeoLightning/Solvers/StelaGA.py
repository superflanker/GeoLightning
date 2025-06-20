"""
Stela GA Wrapper
================

STELA-GA Wrapper - Adaptive Genetic Algorithm for Atmospheric Event Localization

Summary
-------
This module defines a custom wrapper that extends the Genetic Algorithm (GA)
implementation from the MEALPY framework to operate directly on instances of the
`StelaProblem` class, which is designed for lightning geolocation using 
spatio-temporal data.

Before each evolutionary iteration, the search space is dynamically refined
according to the best solution found so far, using the `restart_search_space()` 
method from the problem definition.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- StelaGA class: Genetic Algorithm adapted to STELA geolocation models.

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy  
- mealpy.evolutionary_based.GA  
- GeoLightning.Solvers.StelaProblem
"""

import numpy as np
from mealpy.evolutionary_based.GA import BaseGA
from GeoLightning.Solvers.StelaProblem import StelaProblem


class StelaGA(BaseGA):

    """
    Customized Genetic Algorithm (GA) that integrates STELA's adaptive search
    space refinement mechanism for lightning event localization.

    This class inherits from MEALPY's BaseGA and is modified to work with
    geolocation problems formalized via the `StelaProblem` class.

    Parameters
    ----------
    epoch : int, optional
        Maximum number of iterations. Default is 1000.
    pop_size : int, optional
        Number of individuals in the population. Default is 50.
    **kwargs : dict
        Additional keyword arguments for the base optimizer.
    """

    def evolve(self, pop=None):
        
        """
        Performs one iteration of the Genetic Algorithm with adaptive
        search space refinement.

        This method overrides the default `evolve` behavior by first
        invoking the `restart_search_space()` method of the problem
        to dynamically reduce the search region based on the current best solution.

        Parameters
        ----------
        pop : list of Agent, optional
            The current population of candidate solutions. If not provided,
            the internal population is used.

        Raises
        ------
        TypeError
            If the associated problem is not an instance of `StelaProblem`.

        Returns
        -------
        tuple
            Updated population and the best global solution.
        """
        
        if not isinstance(self.problem, StelaProblem):
            raise TypeError("O problema fornecido deve ser uma instância de StelaProblem.")
        self.problem.restart_search_space()
        super().evolve(pop)
