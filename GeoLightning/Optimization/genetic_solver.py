"""
    EELT 7019 - Inteligência Artificial Aplicada
    Função Fitness - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from typing import List, Dict, Tuple, Union
from mealpy.evolutionary_based import GA
from mealpy import FloatVar
from GeoLightning.Optimization.fitness import fitness
import numpy as np


def run_ga(detections: List[Dict[str, Union[int, float]]],
           sensors: List[Tuple[float, float]],
           num_events: int,
           epochs: int = 100,
           pop_size: int = 50
           ) -> Tuple[np.ndarray, float]:
    """
    Otimiza as localizações e tempos dos eventos, além do agrupamento das detecções,
    utilizando um Algoritmo Genético (GA) da biblioteca Mealpy.

    Args:
        detections (List[Dict[str, Union[int, float]]]): Lista de detecções contendo 'sensor_id' e 'timestamp'.
        sensors (List[Tuple[float, float]]): Lista de sensores [(latitude, longitude)].
        num_events (int): Número máximo estimado de eventos.
        epochs (int, optional): Número de gerações. Default é 100.
        pop_size (int, optional): Tamanho da população. Default é 50.

    Returns:
        Tuple[np.ndarray, float]: 
            - O vetor da melhor solução encontrada (contendo localização, tempo e agrupamento).
            - O valor da função de fitness associado à melhor solução.
    """
    if num_events <= 0:
        raise ValueError("O número de eventos deve ser maior que zero.")
    if not detections:
        raise ValueError("A lista de detecções não pode estar vazia.")
    if not sensors:
        raise ValueError("A lista de sensores não pode estar vazia.")

    num_detections = len(detections)

    lb = [[-50, -50]  for _ in range(num_detections)] 
    ub = [[50, 50] for _ in range(num_detections)] 

    def problem_f(solution):

        # return 100
        return fitness(solution, detections, sensors, num_events)

    problem = {
        "obj_func": problem_f,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "min",
    }

    model = GA.BaseGA(epoch=epochs, pop_size=pop_size)
    best_pos, best_fit = model.solve(problem)
    return best_pos, best_fit
