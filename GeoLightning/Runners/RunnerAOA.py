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
    A tuple containing the following performance indicators:

    centroides_espaciais : np.ndarray
        Array of estimated spatial centroids for each event cluster.
    centroides_temporais : np.ndarray
        Array of estimated origin times (temporal centroids) per cluster.
    detectores : np.ndarray
        Indices of sensors used to estimate temporal centroids.
    best_fitness : float
        Value of the fitness function at the best solution found by the optimizer.
    erro_relativo_best_fitness : float
        Relative error between the best fitness and the reference value.
    rmse_espacial : float
        Root Mean Square Error between estimated and true spatial locations.
    amse_espacial : float
        Average Mean Squared Error between estimated and true spatial locations.
    mle_espacial : float
        Mean Location Error between estimated and true positions.
    prmse_espacial : float
        Pseudo-RMSE normalized by 6 times the spatial noise standard deviation.
    acuracia_associacao_atual : float
        Percentage of correct associations between estimated and true cluster labels.
    tempo_execucao : float
        Total runtime of the optimization process (in seconds).
    crlb_temporal : float
        Cramér-Rao Lower Bound for the origin time estimation.
    rmse_crlb : float
        RMSE corresponding to the spatial CRLB.
    mean_crlb : float
        Mean spatial CRLB across all sensors.
"""

import numpy as np
from GeoLightning.Solvers.StelaProblem import StelaProblem
from GeoLightning.Solvers.StelaAOA import StelaAOA
from GeoLightning.Stela.Stela import stela
from GeoLightning.Stela.Bounds import gera_limites_iniciais
from GeoLightning.Stela.Common import calcular_centroides_espaciais, \
    calcula_distancias_ao_centroide, \
    calcula_residuos_temporais, \
    calcula_centroides_temporais
from GeoLightning.Simulator.Metrics import *
from GeoLightning.Simulator.Simulator import *
from GeoLightning.Utils.Constants import *
from mealpy import FloatVar
from time import perf_counter


def runner_aoa(event_positions: np.ndarray,
               event_times: np.ndarray,
               n_event_positions: np.ndarray,
               n_event_times: np.ndarray,
               detections: np.ndarray,
               detection_times: np.ndarray,
               sensors: np.ndarray,
               min_alt: np.float64,
               max_alt: np.float64,
               min_pts: np.int32 = CLUSTER_MIN_PTS,
               sigma_t: np.float64 = SIGMA_T,
               sigma_d: np.float64 = SIGMA_D,
               epsilon_t: np.float64 = EPSILON_D,
               c: np.float64 = AVG_LIGHT_SPEED,
               sistema_cartesiano: bool = False) -> tuple:

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

    start_st = perf_counter()
    problem = StelaProblem(bounds,
                           minmax="min",
                           pontos_de_chegada=detections,
                           tempos_de_chegada=detection_times,
                           min_pts=min_pts,
                           sigma_t=sigma_t,
                           epsilon_t=epsilon_t,
                           sistema_cartesiano=sistema_cartesiano,
                           c=c)
    model = StelaAOA(epoch=100, pop_size=10)
    agent = model.solve(problem)
    end_st = perf_counter()

    best_solution = agent.solution
    best_fitness = agent.target
    best_solution = np.array(best_solution).reshape(-1, 3)

    # recomputando a clusterização estimada - índice de associação aplicado ao algoritmo

    (clusters_espaciais,
     _) = stela(solucoes=best_solution,
                tempos_de_chegada=detection_times,
                pontos_de_deteccao=detections,
                sistema_cartesiano=sistema_cartesiano,
                epsilon_t=epsilon_t,
                min_pts=min_pts,
                c=c)

    # recomputando a associação real
    (clusters_espaciais_reais,
     f_referencia) = stela(solucoes=n_event_positions,
                           tempos_de_chegada=detection_times,
                           pontos_de_deteccao=detections,
                           sistema_cartesiano=sistema_cartesiano,
                           epsilon_t=epsilon_t,
                           min_pts=min_pts,
                           c=c)

    len_clusterizados = len(
        np.unique(clusters_espaciais[clusters_espaciais >= 0]))
    len_reais = len(event_positions)

    # medições

    # distâncias médias à solução
    centroides_espaciais = calcular_centroides_espaciais(best_solution,
                                                         clusters_espaciais)

    delta_d_real = calcula_distancias_ao_centroide(n_event_times,
                                                   clusters_espaciais,
                                                   event_positions,
                                                   sistema_cartesiano)

    delta_d_estimado = calcula_distancias_ao_centroide(best_solution,
                                                       clusters_espaciais,
                                                       centroides_espaciais,
                                                       sistema_cartesiano)

    # tempos de origem à solução dada pela meta-heurística

    tempos_de_origem = computa_tempos_de_origem(best_solution,
                                                detection_times,
                                                detections,
                                                sistema_cartesiano)

    (centroides_temporais,
     detectores) = calcula_centroides_temporais(tempos_de_origem,
                                                clusters_espaciais)

    delta_t_real = calcula_residuos_temporais(tempos_de_origem,
                                              clusters_espaciais,
                                              event_times)

    delta_t_estimado = calcula_residuos_temporais(n_event_times,
                                                  clusters_espaciais,
                                                  event_times)

    # medidas de distância e tempo

    rmse_espacial = rmse(delta_d_estimado, delta_d_real)

    mae_temporal = mae(delta_t_estimado, delta_t_real)

    amse_espacial = average_mean_squared_error(delta_d_estimado,
                                               delta_d_real)

    mle_espacial = mean_location_error(delta_d_estimado,
                                       delta_d_real)

    prmse_espacial = calcula_prmse(rmse_espacial,
                                   6.0 * SIGMA_D)

    # acurácia de associação

    acuracia_associacao_atual = acuracia_associacao(clusters_espaciais_reais,
                                                    clusters_espaciais)

    # tempo de execução
    tempo_execucao = end_st - start_st

    # porcentagem da função fitness com o valor de referência

    erro_relativo_best_fitness = erro_relativo_funcao_ajuste(best_fitness,
                                                             f_referencia)

    # crlb temporal

    crlb_temporal = calcular_crlb_temporal(sigma_t)

    # crlb espacial

    crlb_espacial = calcular_crlb_espacial(sigma_d, len(detections))

    # rmse crlb espacial

    rmse_crlb = calcular_crlb_rmse(crlb_espacial)

    # mean crlb 

    mean_crlb = calcular_mean_crlb(crlb_espacial)

    return (centroides_espaciais,
            centroides_temporais,
            detectores,
            best_fitness,
            erro_relativo_best_fitness,
            rmse_espacial,
            amse_espacial,
            mle_espacial,
            prmse_espacial,
            acuracia_associacao_atual,
            tempo_execucao, 
            crlb_temporal,
            rmse_crlb,
            mean_crlb)
