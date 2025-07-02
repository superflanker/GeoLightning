"""
EELT 7019 - Applied Artificial Intelligence
Meta-heuristics Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_sensor_matrix,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)

from GeoLightning.Utils.Constants import *
from GeoLightning.Solvers.StelaAOA import StelaAOA
from GeoLightning.Solvers.StelaProblem import StelaProblem
from GeoLightning.Stela.Bounds import gera_limites_iniciais
from GeoLightning.Stela.Stela import stela_phase_one, stela_phase_two
from mealpy import FloatVar
from time import perf_counter


def test_stela_aoa():

    # recuperando o grupo de sensores
    sensors = get_sensors()
    sensor_tt = get_sensor_matrix(sensors, AVG_LIGHT_SPEED, False)
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 935
    max_alt = 935
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
     sensor_indexes,
     spatial_clusters) = generate_detections(event_positions,
                                             event_times,
                                             sensors)

    # limites

    bounds = FloatVar(lb=[min_lat, min_lon, min_alt],
                          ub=[max_lat, max_lon, max_alt])

    # tudo pronto, instanciando a StelaProblem

    problem = StelaProblem(bounds,
                           minmax="min",
                           pontos_de_chegada=detections,
                           tempos_de_chegada=detection_times,
                           sensor_tt=sensor_tt,
                           sensor_indexes=sensor_indexes,
                           min_pts=CLUSTER_MIN_PTS,
                           SIGMA_T=SIGMA_T,
                           epsilon_t=EPSILON_T,
                           sistema_cartesiano=False,
                           c=AVG_LIGHT_SPEED,
                           phase=2)
    
    problem.cluster_it()

    problem_dict = {
        "obj_func": problem.evaluate,  # o próprio objeto como função objetivo
        "bounds": bounds,
        "minmax": "min",
        "n_dims": 3,
        "log_to": None
    }

    start_st = perf_counter()

    model = StelaAOA(epoch=1000,
                     pop_size=50,
                     alpha=3,
                     miu=0.5,
                     moa_min=0.1,
                     moa_max=0.5)
    agent = model.solve(problem_dict)

    end_st = perf_counter()
    print(f"Tempo gasto: {end_st - start_st:.06f}")

    best_solution = agent.solution
    best_fitness = agent.target

    print(best_fitness, best_solution, event_positions)


if __name__ == "__main__":
    test_stela_aoa()
