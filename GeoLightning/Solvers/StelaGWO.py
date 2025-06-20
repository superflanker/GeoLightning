"""
Stela GWO Wrapper
=================

STELA-GWO Wrapper - Grey Wolf Optimizer for Atmospheric Event Localization

Summary
-------
This module defines a custom wrapper that extends the Grey Wolf Optimizer (GWO)
from the MEALPY framework to operate directly on instances of the `StelaProblem`
class, designed for lightning geolocation using spatio-temporal information.

Before each evolutionary iteration, the search space is dynamically refined 
based on the current best solution, using the method `restart_search_space()` 
from the STELA problem instance.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- StelaGWO class: Grey Wolf Optimizer adapted for STELA-based localization.

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy  
- mealpy.swarm_based.GWO  
- GeoLightning.Solvers.StelaProblem
"""

import numpy as np
from mealpy.swarm_based.GWO import OriginalGWO
from GeoLightning.Solvers.StelaProblem import StelaProblem


class StelaGWO(OriginalGWO):

    """
    Customized Grey Wolf Optimizer (GWO) integrated with STELA's
    adaptive geolocation problem formulation.

    This class extends MEALPY's original GWO to operate on dynamic 
    search spaces defined by the `StelaProblem` class, which handles
    lightning geolocation with spatial and temporal constraints.

    Parameters
    ----------
    epoch : int, optional
        Maximum number of iterations. Default is 1000.
    pop_size : int, optional
        Number of individuals in the population. Default is 50.
    **kwargs : dict
        Additional arguments for the base optimizer.
    """

    def evolve(self, epoch):
        """
        Executes one iteration of the algorithm with adaptive search space refinement.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        if not isinstance(self.problem, StelaProblem):
            raise TypeError("The associated problem must be an instance of StelaProblem.")
        self.problem.restart_search_space()
        super().evolve(epoch)
