"""
StelaESO Runner Wrapper
=======================

Summary
-------
Wrapper function for executing the StelaESO algorithm in the context of
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
- Execution of the StelaESO algorithm
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
    - StelaESO
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
from GeoLightning.Solvers.StelaESO import StelaESO
from GeoLightning.Stela.Stela import stela_phase_one, stela_phase_two
from GeoLightning.Stela.Bounds import gera_limites_iniciais
from GeoLightning.Stela.LogLikelihood import maxima_log_verossimilhanca
from GeoLightning.Simulator.Metrics import *
from GeoLightning.Simulator.Simulator import *
from GeoLightning.Utils.Constants import *
from GeoLightning.Utils.Utils import computa_distancia_batelada, computa_distancias
from mealpy import FloatVar
from time import perf_counter


def runner_ESO(event_positions: np.ndarray,
               event_times: np.ndarray,
               spatial_clusters: np.ndarray,
               detections: np.ndarray,
               detection_times: np.ndarray,
               sensors: np.ndarray,
               min_alt: np.float64,
               max_alt: np.float64,
               min_pts: np.int32 = CLUSTER_MIN_PTS,
               sigma_t: np.float64 = SIGMA_T,
               sigma_d: np.float64 = SIGMA_D,
               epsilon_t: np.float64 = EPSILON_T,
               c: np.float64 = AVG_LIGHT_SPEED,
               sistema_cartesiano: bool = False) -> tuple:

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
                           min_pts=min_pts,
                           sigma_t=sigma_t,
                           epsilon_t=epsilon_t,
                           sistema_cartesiano=sistema_cartesiano,
                           c=c,
                           phase=1)
    problem_dict = {
        "obj_func": problem.evaluate,  # o próprio objeto como função objetivo
        "bounds": FloatVar(ub=ub, lb=lb),
        "minmax": "min",
        "n_dims": len(lb),
        "log_to": None
    }
    model = StelaESO(epoch=10,
                     pop_size=10)
    agent = model.solve(problem_dict)

    best_solution = agent.solution
    best_fitness = agent.target.fitness
    best_solution = np.array(best_solution).reshape(-1, 3)

    # recomputando a clusterização estimada - índice de associação aplicado ao algoritmo

    (clusters_espaciais,
     _) = stela_phase_one(solucoes=best_solution,
                          tempos_de_chegada=detection_times,
                          pontos_de_deteccao=detections,
                          sistema_cartesiano=sistema_cartesiano,
                          epsilon_t=epsilon_t,
                          min_pts=min_pts,
                          c=c)

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
                               min_pts=min_pts,
                               sigma_t=sigma_t,
                               epsilon_t=epsilon_t,
                               sistema_cartesiano=sistema_cartesiano,
                               c=c,
                               phase=2)
        problem_dict = {
            "obj_func": problem.evaluate,  # o próprio objeto como função objetivo
            "bounds": bounds,
            "minmax": "min",
            "n_dims": 3,
            "log_to": None
        }

        model = StelaESO(epoch=50,
                         pop_size=25)
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
        sol_reference -= (maxima_log_verossimilhanca(sol_detectores[i], sigma_d) 
                    + maxima_log_verossimilhanca(sol_detectores[i], sigma_t))

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
