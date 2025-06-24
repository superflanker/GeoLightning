"""
HyperParameter Tunning
----------------------

Summary
-------

Tiner Wrapper for PSO, GA and AOA Algorithms

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Submodules
----------
- Tuning.py: Hyperparameter wrapper

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- Mealpy.Tuner
- GeoLighting.Stela
- GeoLightning.Solvers
- GeoLightning.Simulator
"""
import numpy as np
from GeoLightning.Solvers.StelaProblem import StelaProblem
from GeoLightning.Stela.Bounds import gera_limites_iniciais
from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_random_sensors,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)
from GeoLightning.Simulator.Simulator import *
from GeoLightning.Utils.Constants import *
from mealpy import FloatVar, Tuner, Optimizer
from time import perf_counter
from typing import Any, Dict, Union


def tuneit(model: Optimizer,
           param_grid: Dict[str, list],
           term: Dict[str, int]) -> Dict[str, Union[float, Dict[str, Any]]]:
    """
    Perform hyperparameter tuning for a MEALPY-based optimizer on a synthetic lightning event localization problem.

    This function sets up a localization scenario with synthetic atmospheric lightning events and detections
    from a known sensor network, builds a corresponding optimization problem, and applies grid-based
    hyperparameter tuning using the MEALPY Tuner. After executing the trials, it resolves the best configuration
    and returns the optimal hyperparameters and objective value.

    Parameters
    ----------
    model : Optimizer
        An instance of a MEALPY-compatible solver (e.g., LSA, PSO, GA, etc.).
    param_grid : dict of str to list
        Dictionary specifying the hyperparameter grid to explore.
        Each key is a hyperparameter name, and its value is a list of candidate values.
    term : dict of str to int
        Dictionary of stopping criteria passed to the solver, e.g., {"max_epochs": 100}.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'best_score' (float): Best fitness score obtained across all trials.
        - 'best_params' (dict): Dictionary of the best hyperparameter combination found.
    
    Notes
    -----
    - The synthetic scenario includes one lightning event and generates arrival times for a sensor network.
    - The problem is constructed using the StelaProblem class, with parameters such as spatial and temporal tolerances.
    - The tuning is parallelized using threads and includes a post-tuning resolution phase with the best configuration.

    See Also
    --------
    mealpy.Tuner : Interface used for parameter tuning.
    StelaProblem : Custom optimization problem based on TOA geolocation.
    """
    model.logging = False
    # recuperando o grupo de sensores
    sensors = get_sensors()
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
    
    problem_dict = {
        "obj_func": problem.evaluate,  # o próprio objeto como função objetivo
        "bounds": FloatVar(ub=ub, lb=lb),
        "minmax": "min",
        "n_dims": len(lb),
    }
    # semente randomica fixa
    np.random.seed(42)
    # tuner
    tuner = Tuner(model, param_grid)

    tuner.execute(problem=problem_dict,
                  termination=term,
                  n_trials=5,
                  n_jobs=2,
                  mode="single",
                  n_workers=1,
                  verbose=False)

    print(tuner.best_row)

    print(tuner.best_score)

    print(tuner.best_params)
    
    return {"best_score": tuner.best_score,
            "best_params": tuner.best_params}
