"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes Unitários - Função de Otimização - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

import numpy as np
from GeoLightning.Optimization.genetic_solver import run_ga
from GeoLightning.Simulator.simulator import generate_sensors, generate_events
from GeoLightning.Simulator.detection_generator import generate_detections

def test_run_ga_basic_execution():
    """Teste básico: se a função roda e retorna os tipos corretos."""
    sensors = generate_sensors(num_sensors=10)
    events = generate_events(num_events=10)
    detections = generate_detections(events, sensors)

    best_pos, best_fit = run_ga(detections, sensors, num_events=10, epochs=20, pop_size=20)
    print(best_pos)
    assert isinstance(best_pos, np.ndarray), "O best_pos deve ser um np.ndarray."
    assert isinstance(best_fit, float), "O best_fit deve ser float."
    assert best_pos.size == (3 * 4 + len(detections)), "Tamanho esperado do vetor de solução."


def test_run_ga_with_minimal_case():
    """Teste de caso mínimo: 1 evento, 2 sensores, 2 detecções."""
    sensors = [(-10.0, -50.0), (-10.5, -50.5)]
    events = [(-10.2, -50.2, 1.0)]
    detections = generate_detections(events, sensors)

    best_pos, best_fit = run_ga(detections, sensors, num_events=1, epochs=20, pop_size=10)

    assert best_pos is not None
    assert best_fit >= 0
    assert best_pos.size == (4 + len(detections))  # 4 parâmetros do evento + agrupamento


def test_run_ga_raises_on_invalid_num_events():
    """Teste se levanta erro quando num_events <= 0."""
    sensors = [(-10.0, -50.0)]
    events = [(-10.2, -50.2, 1.0)]
    detections = generate_detections(events, sensors)

    try:
        run_ga(detections, sensors, num_events=0)
    except ValueError as e:
        assert str(e) == "O número de eventos deve ser maior que zero."
    else:
        assert False, "Deveria ter levantado ValueError com num_events=0."


def test_run_ga_raises_on_empty_detections():
    """Teste se levanta erro com lista de detecções vazia."""
    sensors = [(-10.0, -50.0)]

    try:
        run_ga([], sensors, num_events=1)
    except ValueError as e:
        assert str(e) == "A lista de detecções não pode estar vazia."
    else:
        assert False, "Deveria ter levantado ValueError com detecções vazias."


def test_run_ga_raises_on_empty_sensors():
    """Teste se levanta erro com lista de sensores vazia."""
    events = [(-10.2, -50.2, 1.0)]
    detections = [{"event_id": 0, "sensor_id": 0, "timestamp": 1.0}]

    try:
        run_ga(detections, [], num_events=1)
    except ValueError as e:
        assert str(e) == "A lista de sensores não pode estar vazia."
    else:
        assert False, "Deveria ter levantado ValueError com sensores vazios."
