"""
Runnr Wrapper for StelaPSO
--------------------------

Summary
-------
Conveniente Wrapper for the StelaPSO algorithm

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- StelaPSO Wrapper

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, 
Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numpy.linalg.inv
- numba
- GeoLightning.Utils.Constants (SIGMA_D, SIGMA_T)
"""
import numpy as np
from GeoLightning.Solvers.StelaProblem import StelaProblem
from GeoLightning.Solvers.StelaPSO import StelaPSO
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


def runner_pso(event_positions: np.ndarray,
               event_times: np.ndarray,
               detections: np.ndarray,
               detection_times: np.ndarray,
               sensors: np.ndarray,
               min_alt: np.float64,
               max_alt: np.float64,
               min_pts: np.int32 = CLUSTER_MIN_PTS,
               sigma_t: np.float64 = SIGMA_T,
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

    problem = StelaProblem(bounds,
                           minmax="min",
                           pontos_de_chegada=detections,
                           tempos_de_chegada=detection_times,
                           min_pts=min_pts,
                           sigma_t=sigma_t,
                           epsilon_t=epsilon_t,
                           sistema_cartesiano=sistema_cartesiano,
                           c=c)

    model = StelaPSO(epoch=100, pop_size=10)
    agent = model.solve(problem)
    best_solution = agent.solution
    best_fitness = agent.target
    best_solution = np.array(best_solution).reshape(-1, 3)

    # recomputando a clusterização - índice de associação aplicado ao algoritmo
    (clusters_espaciais,
     verossimilhanca) = stela(solucoes=best_solution,
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

    delta_d = calcula_distancias_ao_centroide(best_solution,
                                              clusters_espaciais,
                                              event_positions,
                                              sistema_cartesiano)

    # tempos de origem à solução dada pela meta-heurística

    tempos_de_origem = computa_tempos_de_origem(best_solution,
                                                detection_times,
                                                detections,
                                                sistema_cartesiano)

    _, detectores = calcula_centroides_temporais(tempos_de_origem,
                                                 clusters_espaciais)

    delta_t = calcula_residuos_temporais(tempos_de_origem,
                                         clusters_espaciais,
                                         event_times)
