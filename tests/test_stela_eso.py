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
from GeoLightning.Solvers.StelaESO import StelaESO
from GeoLightning.Solvers.StelaProblem import StelaProblem
from GeoLightning.Stela.Bounds import gera_limites_iniciais
from mealpy import FloatVar

def test_stela_eso():

    # recuperando o grupo de sensores
    sensors = get_sensors()
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 935
    max_alt = 935
    min_time = 10000
    max_time = 12100
    num_events = 2000

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
                           sigma_d=SIGMA_D,
                           epsilon_d=EPSILON_D,
                           epsilon_t=EPSILON_T,
                           limit_d=LIMIT_D,
                           max_d=MAX_DISTANCE,
                           sistema_cartesiano=False)
    
    model = StelaESO(epoch=100, pop_size=10)
    agent = model.solve(problem)
    best_solution = agent.solution
    best_fitness = agent.target
    print(best_fitness, best_solution)
    print(problem.clusters_espaciais)
    print(problem.lb)
test_stela_eso()