"""
StelaAOA Runner Wrapper
=======================

Summary
-------
Wrapper function for executing the StelaAOA algorithm in the context of
spatio-temporal event geolocation using lightning detection data. This routine 
initializes the optimization problem, runs the solver, performs the clustering 
of the estimated solutions, and compares results with the reference ground-truth data.

This wrapper is part of the evaluation pipeline for assessing optimization-based
localization strategies, using realistic detection data and metrics aligned with 
Time-of-Arrival (TOA) localization.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- Problem initialization (bounds and detection parameters)
- Execution of the StelaAOA algorithm
- Spatial and temporal cluster assignment via STELA
- Metric evaluation: RMSE, MAE, AMSE, PRMSE, MLE
- CRLB estimates for spatial and temporal precision
- Association accuracy computation and runtime analysis

Notes
-----
This module is part of the academic activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, 
Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
- mealpy
- GeoLightning modules:
    - StelaProblem
    - StelaAOA
    - stela clustering
    - metric evaluation utilities
    - bounding box estimation
    - physical constants (SIGMA_D, SIGMA_T, AVG_LIGHT_SPEED)

Returns
-------
tuple
    sol_centroides_espaciais : np.ndarray
        Array of estimated spatial centroids for each event cluster.
    sol_centroides_temporais : np.ndarray
        Array of estimated origin times (temporal centroids) per cluster.
    sol_detectores : np.ndarray
        Indices of sensors used to estimate temporal centroids.
    sol_best_fitness : float
        Value of the fitness function at the best solution found by the optimizer.
    sol_reference : float
        Value of the fitness function with deltas equal to zero.
    delta_d: np.ndarray
        distance differences between real and estimated positions
    delta_t: np.ndarray
        time differentes between reak and estimated times of origins
    execution_time: np.float64
        total execution time
    associacoes_corretas: np.ndarray
        the correct  clustering association index
"""

import numpy as np
from GeoLightning.Solvers.StelaProblem import StelaProblem
from GeoLightning.Solvers.StelaAOA import StelaAOA
from GeoLightning.Stela.Stela import stela_phase_one, stela_phase_two
from GeoLightning.Stela.Bounds import gera_limites_iniciais
from GeoLightning.Stela.LogLikelihood import maxima_log_verossimilhanca
from GeoLightning.Simulator.Metrics import *
from GeoLightning.Simulator.Simulator import *
from GeoLightning.Utils.Constants import *
from GeoLightning.Utils.Utils import computa_distancia_batelada, computa_distancias
from mealpy import FloatVar
from time import perf_counter


def runner_AOA(event_positions: np.ndarray,
               event_times: np.ndarray,
               spatial_clusters: np.ndarray,
               sensor_tt: np.ndarray,
               sensor_indexes: np.ndarray,
               detections: np.ndarray,
               detection_times: np.ndarray,
               sensors: np.ndarray,
               min_alt: np.float64,
               max_alt: np.float64,
               max_epochs: np.int32 = 100,
               max_population: np.int32 = 100,
               min_pts: np.int32 = CLUSTER_MIN_PTS,
               sigma_t: np.float64 = SIGMA_T,
               sigma_d: np.float64 = SIGMA_D,
               epsilon_t: np.float64 = EPSILON_T,
               c: np.float64 = AVG_LIGHT_SPEED,
               sistema_cartesiano: bool = False) -> tuple:
    """
    Executes the Arithmetic Optimization Algorithm (AOA) algorithm for estimating 
    the origin positions of events based on clustered detections and arrival time data.

    This function applies the AOA metaheuristic to solve the multilateration problem 
    under spatio-temporal constraints defined by STELA. For each spatial cluster 
    of detections, the algorithm estimates the most likely source location that 
    satisfies both the geometric and temporal criteria.

    Parameters
    ----------
    event_positions : np.ndarray
        Ground-truth event positions (used for evaluation or benchmarking).

    event_times : np.ndarray
        Ground-truth emission times of the events.

    spatial_clusters : np.ndarray
        Array of integer labels assigning each detection to a spatial cluster.

    sensor_tt : np.ndarray
        Precomputed time-of-travel matrix between all sensors (in seconds).

    sensor_indexes : np.ndarray
        Indices of sensors associated with each detection.

    detections : np.ndarray
        Spatial coordinates of each detection (e.g., latitude, longitude, altitude).

    detection_times : np.ndarray
        Timestamps of signal arrivals at each sensor (in seconds).

    sensors : np.ndarray
        Coordinates of all sensor positions in the network.

    min_alt : float
        Minimum allowed altitude for candidate event positions (in meters).

    max_alt : float
        Maximum allowed altitude for candidate event positions (in meters).

    max_epochs : int, optional
        Maximum number of iterations (epochs) for the AOA algorithm. Default is 100.

    max_population : int, optional
        Number of candidate solutions in the AOA population. Default is 100.

    min_pts : int, optional
        Minimum number of detections required to form a valid cluster. Default is CLUSTER_MIN_PTS.

    sigma_t : float, optional
        Standard deviation of the temporal measurement noise (in seconds). Default is SIGMA_T.

    sigma_d : float, optional
        Standard deviation of the spatial measurement noise (in meters). Default is SIGMA_D.

    epsilon_t : float, optional
        Maximum allowable temporal deviation for event validity (in seconds). Default is EPSILON_T.

    c : float, optional
        Signal propagation speed (e.g., speed of light) in m/s. Default is AVG_LIGHT_SPEED.

    sistema_cartesiano : bool, optional
        Whether to convert coordinates to a Cartesian system for processing. Default is False.

    Returns
    -------
    tuple
        A tuple containing the following elements:

        sol_centroides_espaciais : np.ndarray  
            Estimated spatial centroids of each cluster (event locations).

        sol_centroides_temporais : np.ndarray  
            Estimated temporal centroids of each cluster (event emission times).

        sol_detectores : list  
            List of sensor indices associated with each optimized cluster.

        sol_best_fitness : np.ndarray  
            Best fitness value obtained by AOA for each cluster.

        sol_reference : np.ndarray  
            Reference fitness value (ground-truth-based) for each cluster.

        delta_d : np.ndarray  
            Spatial deviation (in meters) between estimated and true positions.

        delta_t : np.ndarray  
            Temporal deviation (in seconds) between estimated and true emission times.

        execution_time : float  
            Total time taken to execute the optimization routine (in seconds).

        associacoes_corretas : int  
            Number of clusters correctly associated with ground-truth events.
    """

    start_st = perf_counter()

    # Fase 1: clusterização

    execution_time = 0.0

    # limites

    (min_lat,
     max_lat,
     min_lon,
     max_lon) = get_lightning_limits(sensors)

    lb, ub = gera_limites_iniciais(detections,
                                   min_lat,
                                   max_lat,
                                   min_lon,
                                   max_lon,
                                   min_alt,
                                   max_alt)

    bounds = FloatVar(ub=ub, lb=lb)

    problem = StelaProblem(bounds,
                           minmax="min",
                           pontos_de_chegada=detections,
                           tempos_de_chegada=detection_times,
                           sensor_tt=sensor_tt,
                           sensor_indexes=sensor_indexes,
                           min_pts=min_pts,
                           sigma_t=sigma_t,
                           epsilon_t=epsilon_t,
                           sistema_cartesiano=sistema_cartesiano,
                           c=c)

    problem.cluster_it()

    clusters_espaciais = problem.spatial_clusters

    associacoes_corretas = (clusters_espaciais == spatial_clusters)
    corretos = associacoes_corretas[associacoes_corretas == True]
    print(
        f"Média de Eventos Clusterizados Corretamente: {100 * len(corretos)/len(spatial_clusters):.04f} % ")

    # Fase 2 - Refinamento da Solução

    max_clusters = np.max(clusters_espaciais) + 1

    sol_centroides_espaciais = np.empty(
        (max_clusters, event_positions.shape[1]))

    sol_centroides_temporais = np.empty(max_clusters)

    sol_detectores = np.empty(max_clusters)

    sol_best_fitness = 0.0

    sol_reference = 0.0

    for i in range(max_clusters):

        current_detections = np.array(detections[clusters_espaciais == i])

        current_detection_times = np.array(
            detection_times[clusters_espaciais == i])

        detectores = len(current_detection_times)

        bounds = FloatVar(lb=[min_lat, min_lon, min_alt],
                          ub=[max_lat, max_lon, max_alt])

        problem = StelaProblem(bounds,
                               minmax="min",
                               pontos_de_chegada=current_detections,
                               tempos_de_chegada=current_detection_times,
                               sensor_tt=sensor_tt,
                               sensor_indexes=sensor_indexes,
                               min_pts=min_pts,
                               sigma_t=sigma_t,
                               epsilon_t=epsilon_t,
                               sistema_cartesiano=sistema_cartesiano,
                               c=c)

        problem.spatial_clusters = np.zeros(
            len(current_detections), dtype=np.int32)

        problem_dict = {
            "obj_func": problem.evaluate,  # o próprio objeto como função objetivo
            "bounds": bounds,
            "minmax": "min",
            "n_dims": 3,
            "log_to": None
        }

        model = StelaAOA(epoch=max_epochs,
                         pop_size=max_population,
                         alpha=4,
                         miu=0.5,
                         moa_min=0.1,
                         moa_max=0.8)
        agent = model.solve(problem_dict)

        best_solution = agent.solution
        best_fitness = agent.target.fitness

        o_distancias = computa_distancias(best_solution,
                                          current_detections,
                                          sistema_cartesiano)

        tempos_de_origem = current_detection_times - o_distancias/AVG_LIGHT_SPEED

        centroide_temporal = np.mean(tempos_de_origem)

        # não preciso ter medo pois é um cluster somente
        sol_centroides_espaciais[i] = best_solution
        # não preciso ter medo pois é um cluster somente
        sol_centroides_temporais[i] = centroide_temporal
        sol_detectores[i] = detectores

        # valores para calcular o erro relativo em relação ao valor de referência
        sol_best_fitness += np.abs(best_fitness)
        sol_reference -= maxima_log_verossimilhanca(sol_detectores[i], sigma_d)

    # medições

    # tempos de origem à solução dada pela meta-heurística

    delta_d = computa_distancia_batelada(sol_centroides_espaciais,
                                         event_positions)

    delta_t = event_times - sol_centroides_temporais

    end_st = perf_counter()

    execution_time = end_st - start_st

    return (sol_centroides_espaciais,
            sol_centroides_temporais,
            sol_detectores,
            sol_best_fitness,
            sol_reference,
            delta_d,
            delta_t,
            execution_time,
            associacoes_corretas)
