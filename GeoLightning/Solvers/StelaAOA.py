"""
Stela AOA Wrapper
=================

STELA-AOA Wrapper - Adaptive Optimization for Lightning Event Localization

Summary
-------
This module defines a wrapper class that extends the Arithmetic Optimization Algorithm (AOA)
from the MEALPY framework to directly operate on instances of the `StelaProblem` class.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- StelaAOA class: customized AOA with adaptive search space refinement.
- Integration with STELA-based geolocation problems.

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- mealpy
- GeoLightning.Solvers.StelaProblem
"""

import numpy as np
from mealpy.math_based.AOA import OriginalAOA


class StelaAOA(OriginalAOA):

    """
    A customized version of the Arithmetic Optimization Algorithm (AOA) 
    designed to solve geolocation problems formulated as `StelaProblem`.

    This class overrides the standard AOA behavior by adaptively refining 
    the search space before each evolutionary step using the problem's 
    internal logic.

    Parameters
    ----------
    epoch : int, optional
        Number of evolutionary iterations (default is 1000).
    pop_size : int, optional
        Number of candidate solutions (default is 50).
    **kwargs : dict, optional
        Additional arguments passed to the parent class.
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
