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


class StelaGWO(OriginalGWO):

    """
    Customized Grey Wolf Optimizer (GWO) integrated with STELA's
    adaptive geolocation problem formulation.

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
