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

from GeoLightning.Stela.STDBSCAN import st_dbscan
from GeoLightning.Utils.Constants import *
from GeoLightning.Solvers.StelaPSO import StelaPSO
from GeoLightning.Solvers.StelaProblem import StelaProblem
from GeoLightning.Stela.Bounds import gera_limites_iniciais
from mealpy import FloatVar

def test_stela_pso():

    # recuperando o grupo de sensores
    sensors = get_sensors()
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 0
    max_alt = 10000
    min_time = 10000
    max_time = 10100
    num_events = 100

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

    ub, lb = gera_limites_iniciais(detections,
                                   min_lat, 
                                   max_lat, 
                                   min_lon, 
                                   max_lon, 
                                   min_alt, 
                                   max_alt)

    bounds = FloatVar(ub=ub, lb=lb)

    # tudo pronto, instanciando a StelaProblem

    problem = StelaProblem(bounds, "max", detections, detection_times)
    model = StelaPSO(epoch=100, pop_size=50)
    best_solution, best_fitness = model.solve(problem)