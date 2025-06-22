"""
Stela AOA Wrapper
=================

STELA-AOA Wrapper - Adaptive Optimization for Lightning Event Localization

Summary
-------
This module defines a wrapper class that extends the Arithmetic Optimization Algorithm (AOA)
from the MEALPY framework to directly operate on instances of the `StelaProblem` class.

The optimization process dynamically refines the search space boundaries based on 
the best solution found so far, integrating a feedback loop via the 
`restart_search_space()` method of the problem.

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
from GeoLightning.Solvers.StelaProblem import StelaProblem


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
        if not isinstance(self.problem, StelaProblem):
            raise TypeError("The associated problem must be an instance of StelaProblem.")
        super().evolve(epoch)
