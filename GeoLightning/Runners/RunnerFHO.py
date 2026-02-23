"""
StelaFHO Runner Wrapper
=======================

Summary
-------
Wrapper function for executing the StelaFHO algorithm in the context of
spatio-temporal event geolocation using lightning detection data. This routine 
initializes the optimization problem, runs the solver, performs the clustering 
of the estimated solutions, and compares results with the reference ground-truth data.

This wrapper is part of the evaluation pipeline for assessing optimization-based
localization strategies, using realistic detection data and metrics aligned with 
Time-of-Arrival (TOA) localization.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- Problem initialization (bounds and detection parameters)
- Execution of the StelaFHO algorithm
- Spatial and temporal cluster assignment via STELA
- Metric evaluation: RMSE, MAE, AMSE, PRMSE, MLE
- CRLB estimates for spatial and temporal precision
- Association accuracy computation and runtime analysis

Notes
-----
This module is part of the academic activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, 
Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
- mealpy
- GeoLightning modules:
    - StelaProblem
    - StelaFHO
    - stela clustering
    - metric evaluation utilities
    - bounding box estimation
    - physical constants (SIGMA_D, SIGMA_T, AVG_LIGHT_SPEED)

Returns
-------
tuple
    sol_centroides_espaciais : np.ndarray
        Array of estimated spatial centroids for each event cluster.
    sol_centroides_temporais : np.ndarray
        Array of estimated origin times (temporal centroids) per cluster.
    sol_detectores : np.ndarray
        Indices of sensors used to estimate temporal centroids.
    sol_best_fitness : float
        Value of the fitness function at the best solution found by the optimizer.
    sol_reference : float
        Value of the fitness function with deltas equal to zero.
    delta_d: np.ndarray
        distance differences between real and estimated positions
    delta_t: np.ndarray
        time differentes between reak and estimated times of origins
    execution_time: np.float64
        total execution time
    associacoes_corretas: np.ndarray
        the correct  clustering association index
"""

import numpy as np
from GeoLightning.Runners.Runner import runner
from GeoLightning.Solvers.StelaFHO import StelaFHO
from GeoLightning.Utils.Constants import *


def runner_FHO(event_positions: np.ndarray,
               event_times: np.ndarray,
               spatial_clusters: np.ndarray,
               sensor_tt: np.ndarray,
               sensor_indexes: np.ndarray,
               detections: np.ndarray,
               detection_times: np.ndarray,
               sensors: np.ndarray,
               min_alt: np.float64,
               max_alt: np.float64,
               max_epochs: np.int32 = 100,
               max_population: np.int32 = 100,
               min_pts: np.int32 = CLUSTER_MIN_PTS,
               sigma_t: np.float64 = SIGMA_T,
               sigma_d: np.float64 = SIGMA_D,
               epsilon_t: np.float64 = EPSILON_T,
               c: np.float64 = AVG_LIGHT_SPEED,
               sistema_cartesiano: bool = False) -> tuple:
    """
    Executes the Fire Hawk Optimizer (FHO) algorithm for estimating 
    the origin positions of events based on clustered detections and arrival time data.

    This function applies the FHO metaheuristic to solve the multilateration problem 
    under spatio-temporal constraints defined by STELA. For each spatial cluster 
    of detections, the algorithm estimates the most likely source location that 
    satisfies both the geometric and temporal criteria.

    Parameters
    ----------
    event_positions : np.ndarray
        Ground-truth event positions (used for evaluation or benchmarking).

    event_times : np.ndarray
        Ground-truth emission times of the events.

    spatial_clusters : np.ndarray
        Array of integer labels assigning each detection to a spatial cluster.

    sensor_tt : np.ndarray
        Precomputed time-of-travel matrix between all sensors (in seconds).

    sensor_indexes : np.ndarray
        Indices of sensors associated with each detection.

    detections : np.ndarray
        Spatial coordinates of each detection (e.g., latitude, longitude, altitude).

    detection_times : np.ndarray
        Timestamps of signal arrivals at each sensor (in seconds).

    sensors : np.ndarray
        Coordinates of all sensor positions in the network.

    min_alt : float
        Minimum allowed altitude for candidate event positions (in meters).

    max_alt : float
        Maximum allowed altitude for candidate event positions (in meters).

    max_epochs : int, optional
        Maximum number of iterations (epochs) for the FHO algorithm. Default is 100.

    max_population : int, optional
        Number of candidate solutions in the FHO population. Default is 100.

    min_pts : int, optional
        Minimum number of detections required to form a valid cluster. Default is CLUSTER_MIN_PTS.

    sigma_t : float, optional
        Standard deviation of the temporal measurement noise (in seconds). Default is SIGMA_T.

    sigma_d : float, optional
        Standard deviation of the spatial measurement noise (in meters). Default is SIGMA_D.

    epsilon_t : float, optional
        Maximum allowable temporal deviation for event validity (in seconds). Default is EPSILON_T.

    c : float, optional
        Signal propagation speed (e.g., speed of light) in m/s. Default is AVG_LIGHT_SPEED.

    sistema_cartesiano : bool, optional
        Whether to convert coordinates to a Cartesian system for processing. Default is False.

    Returns
    -------
    tuple
        A tuple containing the following elements:

        sol_centroides_espaciais : np.ndarray  
            Estimated spatial centroids of each cluster (event locations).

        sol_centroides_temporais : np.ndarray  
            Estimated temporal centroids of each cluster (event emission times).

        sol_detectores : list  
            List of sensor indices associated with each optimized cluster.

        sol_best_fitness : np.ndarray  
            Best fitness value obtained by FHO for each cluster.

        sol_reference : np.ndarray  
            Reference fitness value (ground-truth-based) for each cluster.

        delta_d : np.ndarray  
            Spatial deviation (in meters) between estimated and true positions.

        delta_t : np.ndarray  
            Temporal deviation (in seconds) between estimated and true emission times.

        execution_time : float  
            Total time taken to execute the optimization routine (in seconds).

        associacoes_corretas : int  
            Number of clusters correctly associated with ground-truth events.
    """

    solver = StelaFHO(epoch=max_epochs,
                      pop_size=max_population)

    return runner(solver=solver,
                  event_positions=event_positions,
                  event_times=event_times,
                  spatial_clusters=spatial_clusters,
                  sensor_tt=sensor_tt,
                  sensor_indexes=sensor_indexes,
                  detections=detections,
                  detection_times=detection_times,
                  sensors=sensors,
                  min_alt=min_alt,
                  max_alt=max_alt,
                  min_pts=min_pts,
                  sigma_t=sigma_t,
                  sigma_d=sigma_d,
                  epsilon_t=epsilon_t,
                  c=c,
                  sistema_cartesiano=sistema_cartesiano)


def runner_FHO_process(params):

    event_positions = params["event_positions"]
    event_times = params["event_times"]
    spatial_clusters = params["spatial_clusters"]
    sensor_tt = params["sensor_tt"]
    sensor_indexes = params["sensor_indexes"]
    detections = params["detections"]
    detection_times = params["detection_times"]
    sensors = params["sensors"]
    min_alt = params["min_alt"]
    max_alt = params["max_alt"]
    max_epochs = params["max_epochs"]
    max_population = params["max_population"]
    min_pts = params["min_pts"]
    sigma_t = params["sigma_t"]
    sigma_d = params["sigma_d"]
    epsilon_t = params["epsilon_t"]
    c = params["c"]
    sistema_cartesiano = params["sistema_cartesiano"]

    return runner_FHO(event_positions=event_positions,
                      event_times=event_times,
                      spatial_clusters=spatial_clusters,
                      sensor_tt=sensor_tt,
                      sensor_indexes=sensor_indexes,
                      detections=detections,
                      detection_times=detection_times,
                      sensors=sensors,
                      min_alt=min_alt,
                      max_alt=max_alt,
                      max_epochs=max_epochs,
                      max_population=max_population,
                      min_pts=min_pts,
                      sigma_t=sigma_t,
                      sigma_d=sigma_d,
                      epsilon_t=epsilon_t,
                      c=c,
                      sistema_cartesiano=sistema_cartesiano)
