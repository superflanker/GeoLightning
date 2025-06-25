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
