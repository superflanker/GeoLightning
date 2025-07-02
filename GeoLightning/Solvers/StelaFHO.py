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
        Executes one iteration of the algorithm.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        """if not isinstance(self.problem, StelaProblem):
            raise TypeError(
                "The associated problem must be an instance of StelaProblem.")"""
        super().evolve(epoch)

    def amend_solution(self, position: np.ndarray) -> np.ndarray:
        """
        Clamp each dimension of the solution vector to lie within its allowed bounds.

        Parameters
        ----------
        position : np.ndarray
            The candidate solution vector to be amended.

        Returns
        -------
        np.ndarray
            The amended solution vector with values within bounds.
        """
        lower_bounds = self.problem.lb
        upper_bounds = self.problem.ub
        return np.clip(position, lower_bounds, upper_bounds)

    def correct_position(self, position: np.ndarray) -> np.ndarray:
        """
        Clamp the position vector to the feasible domain using bound limits.

        Parameters
        ----------
        position : np.ndarray
            Candidate solution vector to correct.

        Returns
        -------
        np.ndarray
            Corrected position vector within bounds.
        """
        return self.amend_position(position)
