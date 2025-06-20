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
from GeoLightning.Utils.Constants import SIGMA_D, \
    EPSILON_D, \
    EPSILON_T, \
    LIMIT_D, \
    CLUSTER_MIN_PTS, \
    MAX_DISTANCE


class StelaProblem(Problem):
    def __init__(self,
                 bounds,
                 minmax,
                 pontos_de_chegada: np.ndarray,
                 tempos_de_chegada: np.ndarray,
                 sistema_cartesiano: bool = False,
                 sigma_d: np.float64 = SIGMA_D,
                 epsilon_t: np.float64 = EPSILON_T,
                 epsilon_d: np.float64 = EPSILON_D,
                 limit_d: np.float64 = LIMIT_D,
                 max_d: np.float64 = MAX_DISTANCE,
                 min_pts: np.int32 = CLUSTER_MIN_PTS,
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
        sigma_d : float, optional
            Standard deviation of distance measurement error.
        epsilon_t : float, optional
            Maximum temporal tolerance for clustering.
        epsilon_d : float, optional
            Maximum spatial tolerance for clustering.
        limit_d : float, optional
            Radius for local refinement during optimization.
        max_d : float, optional
            Maximum admissible distance between events and detections.
        min_pts : int, optional
            Minimum number of detections for a cluster to be considered valid.
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
        self.sigma_d = sigma_d
        self.epsilon_d = epsilon_d
        self.epsilon_t = epsilon_t
        self.limit_d = limit_d
        self.max_d = max_d
        self.min_pts = min_pts

        # variáveis internas de controle
        self.fitness_values = list()
        self.stela_ub = list()
        self.stela_lb = list()
        self.stela_centroides = list()
        self.stela_clusters_espaciais = list()
        self.stela_novas_solucoes = list()
        self.stela_detectores = list()
        # variáveis de resposta
        self.clusters_espaciais = - \
            np.ones(pontos_de_chegada.shape[0], dtype=np.int32)
        self.centroides = -np.ones(pontos_de_chegada.shape)
        self.detectores = -np.ones(pontos_de_chegada.shape[0], dtype=np.int32)
        super().__init__(bounds, minmax, solution_encoding="float", **kwargs)

    def restart_search_space(self):
        """
        Dynamically refines the search space based on the best solution 
        encountered during the evolutionary process.

        Updates the internal upper (`self.ub`) and lower (`self.lb`) bounds
        using the highest scoring solution (according to `minmax`). This method 
        is intended to iteratively focus the search around more promising regions.

        Side Effects
        ------------
        - Updates the internal search bounds (`self.lb`, `self.ub`).
        - Updates the attributes `clusters_espaciais`, `centroides`, `detectores`.
        - Clears all temporary storage used in the previous evaluation cycle.
        """
        if len(self.fitness_values) > 0:
            fitness_values = np.array(self.fitness_values)
            # encontrando a melhor solução dentre as sugeridas
            if self.minmax == "min":
                best_fitness_index = np.argwhere(
                    fitness_values == np.min(fitness_values)).flatten()[0]
            else:
                best_fitness_index = np.argwhere(
                    fitness_values == np.max(-fitness_values)).flatten()[0]
            # ajustando os limites 
            bounds = FloatVar(ub=self.stela_ub[best_fitness_index],
                              lb=self.stela_lb[best_fitness_index])
            self.set_bounds(bounds)
            # guardando informações finais
            self.clusters_espaciais = self.stela_clusters_espaciais[best_fitness_index]
            self.centroides = self.stela_centroides[best_fitness_index]
            self.detectores = self.stela_detectores[best_fitness_index]
            # reiniciando as listas
            self.fitness_values = list()
            self.stela_ub = list()
            self.stela_lb = list()
            self.stela_centroides = list()
            self.stela_clusters_espaciais = list()
            self.stela_novas_solucoes = list()
            self.stela_detectores = list()

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

    def get_best_solution(self):
        """
        Retrieves the best solution and its fitness score found so far.

        Returns
        -------
        tuple
            - np.ndarray: the best solution vector.
            - float: the corresponding fitness value.
        """
        if not self.fitness_values:
            return None, None
        idx = np.argmin(self.fitness_values) if self.minmax == "min" else np.argmax(
            self.fitness_values)

        return self.stela_novas_solucoes[idx], self.fitness_values[idx]

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
        solucoes = self.decode_solution(solution)

        # Executa o algoritmo STELA
        (lb,
         ub,
         centroides,
         detectores,
         clusters_espaciais,
         novas_solucoes,
         verossimilhanca) = stela(solucoes,
                                  self.tempos_de_chegada,
                                  self.pontos_de_chegada,
                                  self.clusters_espaciais,
                                  self.sistema_cartesiano,
                                  self.sigma_d,
                                  self.epsilon_t,
                                  self.epsilon_d,
                                  self.limit_d,
                                  self.max_d,
                                  self.min_pts)

        # Armazena resultados auxiliares para possível refinamento posterior
        self.fitness_values.append(-verossimilhanca if self.minmax ==
                                   "max" else verossimilhanca)
        self.stela_ub.append(ub)
        self.stela_lb.append(lb)
        self.stela_centroides.append(centroides)
        self.stela_clusters_espaciais.append(clusters_espaciais)
        self.stela_novas_solucoes.append(novas_solucoes)
        self.stela_detectores.append(detectores)

        # Retorna a verossimilhança como valor de fitness (negativa para problema
        # de maximização)
        return -verossimilhanca if self.minmax == "max" else verossimilhanca
