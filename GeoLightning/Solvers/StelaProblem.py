"""
Stela Problem Wrapper
=====================

StelaProblem Class - Spatio-Temporal Event Estimation

Summary
-------
This module defines the `StelaProblem` class, which encapsulates a spatio-temporal 
localization problem using time-of-arrival (TOA) information from a distributed 
sensor network. It is suitable for atmospheric event localization, such as lightning, 
using a maximum-likelihood-based fitness function calculated via the STELA algorithm.

Each candidate solution encodes the (lat, lon, alt) coordinates of M potential events, 
and the quality of each solution is evaluated based on spatial and temporal clustering 
likelihood.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- StelaProblem class: problem formulation compatible with MEALPY
- restart_search_space(): dynamically adjusts bounds
- evaluate(): objective function wrapper
- get_best_solution(): retrieves the current best candidate
- obj_func(): spatio-temporal likelihood function based on STELA

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- mealpy
- GeoLightning.Stela.Stela
- GeoLightning.Utils.Constants
"""

import numpy as np
from mealpy import FloatVar, Problem
from GeoLightning.Stela.Stela import stela
from GeoLightning.Utils.Constants import SIGMA_T, \
    EPSILON_T, \
    CLUSTER_MIN_PTS, \
    AVG_LIGHT_SPEED


class StelaProblem(Problem):
    def __init__(self,
                 bounds,
                 minmax,
                 pontos_de_chegada: np.ndarray,
                 tempos_de_chegada: np.ndarray,
                 sistema_cartesiano: bool = False,
                 sigma_t: np.float64 = SIGMA_T,
                 epsilon_t: np.float64 = EPSILON_T,
                 min_pts: np.int32 = CLUSTER_MIN_PTS,
                 c: np.float64 = AVG_LIGHT_SPEED,
                 **kwargs):
        """
        Initialize an instance of the STELA problem for use with MEALPY 
        metaheuristic optimization algorithms.

        Parameters
        ----------
        bounds : list of np.ndarray
            A list containing two arrays [lower_bounds, upper_bounds] of shape (3M,).
            Each triplet (lat, lon, alt) corresponds to one candidate event.
        minmax : str
            Optimization type: "min" or "max".
        pontos_de_chegada : np.ndarray
            Array of shape (N, 3) with sensor positions [latitude, longitude, altitude].
        tempos_de_chegada : np.ndarray
            Arrival times of signals at each sensor (shape: N,).
        sistema_cartesiano : bool, optional
            If True, uses Cartesian coordinates; otherwise, geodetic coordinates.
        sigma_t: float, optional
            Standard deviation of time measurement error.
        epsilon_t : float, optional
            Maximum temporal tolerance for clustering.
        min_pts : int, optional
            Minimum number of detections for a cluster to be considered valid.
        c: float, optional
            wave propagation velocity (Default is average light speed)
        **kwargs : dict
            Additional arguments for the MEALPY `Problem` base class.

        Attributes
        ----------
        clusters_espaciais : np.ndarray
            Spatial cluster labels assigned to each detection.
        centroides : np.ndarray
            Coordinates of event centroids (lat, lon, alt).
        detectores : np.ndarray
            Binary mask indicating whether each sensor was involved in a solution.
        """
        # parâmetros passados
        self.pontos_de_chegada = pontos_de_chegada
        self.tempos_de_chegada = tempos_de_chegada
        self.sistema_cartesiano = sistema_cartesiano
        self.sigma_t = sigma_t
        self.epsilon_t = epsilon_t
        self.min_pts = min_pts
        self.avg_speed = c
        self.verbose = False
        super().__init__(bounds, minmax, **kwargs)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def evaluate(self, solution):
        """
        Evaluates a solution using the defined objective function.

        Parameters
        ----------
        solution : np.ndarray
            A 1D array encoding the flattened coordinates of M candidate events.

        Returns
        -------
        list
            A list with one element containing the objective function value.
        """
        return [self.obj_func(solution)]
    

    def obj_func(self, solution):
        """
        Objective function for the STELA problem.

        Evaluates the spatio-temporal likelihood of a candidate solution, 
        calculated using the STELA algorithm. This algorithm performs clustering
        and refinement using both arrival times and spatial data from sensors.

        Parameters
        ----------
        solution : np.ndarray
            A flat array representing (lat, lon, alt) coordinates for M events.

        Returns
        -------
        float
            The objective value (likelihood). Returns the negative value if
            the problem is a maximization task.
        """
        # Converte o vetor linear para o formato (M, 3)
        # solucoes = self.decode_solution(solution)

        solucoes = np.array(solution.reshape(-1, 3))

        # Executa o algoritmo STELA
        (_,
         verossimilhanca) = stela(solucoes,
                                  self.tempos_de_chegada,
                                  self.pontos_de_chegada,
                                  self.sistema_cartesiano,
                                  self.sigma_t,
                                  self.epsilon_t,
                                  self.min_pts,
                                  self.avg_speed)

        # Retorna a verossimilhança como valor de fitness (negativa para problema
        # de maximização)    
        if self.minmax == "min":
            return -verossimilhanca
        return verossimilhanca
