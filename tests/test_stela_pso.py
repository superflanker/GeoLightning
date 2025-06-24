"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes de Metaheurísticas
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_random_sensors,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)

from GeoLightning.Utils.Constants import *
from GeoLightning.Solvers.StelaPSO import StelaPSO
from GeoLightning.Solvers.StelaProblem import StelaProblem
from GeoLightning.Stela.Bounds import gera_limites_iniciais
from GeoLightning.Stela.Stela import stela
from mealpy import FloatVar
from time import perf_counter

def test_stela_pso():

    # recuperando o grupo de sensores
    sensors = get_sensors()
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 0
    max_alt = 1
    min_time = 10000
    max_time = min_time + 72 * 3600
    num_events = 1

    event_positions, event_times = generate_events(num_events,
                                                   min_lat,
                                                   max_lat,
                                                   min_lon,
                                                   max_lon,
                                                   min_alt,
                                                   max_alt,
                                                   min_time,
                                                   max_time)

    # gerando as detecções
    (detections,
     detection_times,
     n_event_positions,
     n_event_times,
     distances,
     spatial_clusters) = generate_detections(event_positions,
                                             event_times,
                                             sensors)
    
    # limites

    lb, ub = gera_limites_iniciais(detections,
                                   min_lat, 
                                   max_lat, 
                                   min_lon, 
                                   max_lon, 
                                   min_alt, 
                                   max_alt)

    bounds = FloatVar(ub=ub, lb=lb)

    # tudo pronto, instanciando a StelaProblem
    problem = StelaProblem(bounds, 
                           minmax="min", 
                           pontos_de_chegada=detections, 
                           tempos_de_chegada=detection_times,
                           min_pts=CLUSTER_MIN_PTS,
                           SIGMA_T=SIGMA_T,
                           epsilon_t=EPSILON_T,
                           sistema_cartesiano=False)
    
    start_st = perf_counter()

    model = StelaPSO(epoch=100, 
                     pop_size=100, 
                     c1=1.5, 
                     c2=1.5, 
                     w=0.5)
    
    agent = model.solve(problem)
    
    end_st = perf_counter()
    print(f"Tempo gasto: {end_st - start_st:.06f}")

    best_solution = agent.solution
    best_fitness = agent.target
    best_solution = np.array(best_solution).reshape(-1,3)
        # recomputando a clusterização - índice de associação aplicado ao algoritmo
    (clusters_espaciais, 
     verossimilhanca) = stela(solucoes=best_solution,
                              tempos_de_chegada=detection_times,
                              pontos_de_deteccao=detections,
                              sistema_cartesiano=False,
                              epsilon_t=EPSILON_T,
                              min_pts=CLUSTER_MIN_PTS,
                              c=AVG_LIGHT_SPEED)
    len_clusterizados = len(
        np.unique(clusters_espaciais[clusters_espaciais >= 0]))
    len_reais = len(event_positions)
    assert len_clusterizados == len_reais
if __name__ == "__main__":
    test_stela_pso()