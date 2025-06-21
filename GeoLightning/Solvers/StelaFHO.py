"""
Stela FHO Wrapper
=================

STELA-FHO Wrapper - Adaptive Optimization for Lightning Event Localization

Summary
-------
This module defines a wrapper class that extends the Fire Hawk Optimizer (FHO)
from the MEALPY framework to work directly with instances of the `StelaProblem` class.

The optimization process includes adaptive search space refinement using the
`restart_search_space()` method from the STELA geolocation problem formulation.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- StelaFHO class: customized FHO with problem-specific space adaptation.

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy  
- GeoLightning.Solvers.Mealpy.FHO  
- GeoLightning.Solvers.StelaProblem
"""

import numpy as np
from GeoLightning.Solvers.Mealpy.FHO import FHO
from GeoLightning.Solvers.StelaProblem import StelaProblem


class StelaFHO(FHO):

    """
    A customized version of the Fire Hawk Optimizer (FHO) algorithm
    adapted to solve geolocation problems defined as `StelaProblem`.

    This class overrides the default behavior by applying adaptive search
    space refinement prior to each evolutionary step, based on the best
    solution found so far.

    Parameters
    ----------
    epoch : int, optional
        Number of optimization iterations (default is 1000).
    pop_size : int, optional
        Number of candidate solutions (default is 50).
    **kwargs : dict
        Additional arguments passed to the MEALPY base optimizer.
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
