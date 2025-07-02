"""
Stela LSA Wrapper
=================

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


class StelaLSA(LSA):
    """
    Specialized class extending MEALPY's LSA optimizer to work with 
    `StelaProblem` instances for spatio-temporal event localization.
    
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
