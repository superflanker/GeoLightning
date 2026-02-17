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
from GeoLightning.Solvers.StelaLSA import StelaLSA
from GeoLightning.Solvers.StelaProblem import StelaProblem
from GeoLightning.Stela.Bounds import gera_limites_iniciais
from GeoLightning.Stela.Stela import stela_phase_one, stela_phase_two
from mealpy import FloatVar
from time import perf_counter


def test_stela_lsa():

    # recuperando o grupo de sensores

    sensors = get_sensors()

    sensor_tt = get_sensor_matrix(sensors=sensors,
                                  wave_speed=AVG_LIGHT_SPEED,
                                  sistema_cartesiano=False)

    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensores_latlon=sensors,
                                                              margem_metros=3000)

    from scipy.spatial import ConvexHull

    # sensores: array [[lat, lon], ...]
    hull = ConvexHull(sensors[:, :2])
    vertices_hull = sensors[hull.vertices, :2] # Apenas os sensores da borda,

    delta_time = 0.0

    for i in range(len(sensor_tt)):
        for j in range(i, len(sensor_tt[i])):
            if sensor_tt[i, j] > delta_time:
                delta_time = sensor_tt[i, j]

    paper_data = list()

    deltas_d = list()

    deltas_t = list()

    # dados default

    num_events = 1

    runs = 1

    sigma_t = SIGMA_T

    multiplier = 3

    # gerando os eventos
    min_alt = 935
    max_alt = 935
    min_time = 10000
    max_time = min_time + num_events * delta_time * multiplier

    # protagonista da história - eventos
    event_positions, event_times = generate_events(num_events=num_events,
                                                   vertices_hull=vertices_hull,
                                                   min_lat=min_lat,
                                                   max_lat=max_lat,
                                                   min_lon=min_lon,
                                                   max_lon=max_lon,
                                                   min_alt=min_alt,
                                                   max_alt=max_alt,
                                                   min_time=min_time,
                                                   max_time=max_time)

    # gerando as detecções
    (detections,
        detection_times,
        n_event_positions,
        n_event_times,
        distances,
        sensor_indexes,
        spatial_clusters) = generate_detections(event_positions=event_positions,
                                                event_times=event_times,
                                                sensor_positions=sensors,
                                                simulate_complete_detections=True,
                                                fixed_seed=False,
                                                min_pts=CLUSTER_MIN_PTS)
    
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
                           c=AVG_LIGHT_SPEED)
    
    problem.cluster_it()

    problem_dict = {
        "obj_func": problem.evaluate,  # o próprio objeto como função objetivo
        "bounds": bounds,
        "minmax": "min",
        "n_dims": 3,
        "log_to": None
    }
    start_st = perf_counter()

    model = StelaLSA(epoch=150,
                     pop_size=40)
    agent = model.solve(problem_dict)

    end_st = perf_counter()
    print(f"Tempo gasto: {end_st - start_st:.06f}")

    best_solution = agent.solution
    best_fitness = agent.target

    print(best_fitness, best_solution, event_positions)



if __name__ == "__main__":
    test_stela_lsa()
