"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes Unitários - Função Fitness - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

import numpy as np
from GeoLightning.Optimization.fitness import fitness

def test_fitness_basic():
    sensors = [(-10.0, -50.0), (-10.5, -50.5)]
    detections = [
        {"sensor_id": 0, "timestamp": 1.002},
        {"sensor_id": 1, "timestamp": 1.005}
    ]

    solution = np.array([
        -10.2, -50.2, 1.0, 1.0,
        -10.5, -50.5, 5.0, 0.0,
        0, 0
    ])

    score = fitness(solution, detections, sensors, max_events=2)
    assert score < 1.0

def test_fitness_with_invalid_assignment():
    sensors = [(-10.0, -50.0)]
    detections = [{"sensor_id": 0, "timestamp": 1.0}]

    solution = np.array([
        -10.2, -50.2, 1.0, 0.0,
        0
    ])

    score = fitness(solution, detections, sensors, max_events=1)
    assert score >= 10