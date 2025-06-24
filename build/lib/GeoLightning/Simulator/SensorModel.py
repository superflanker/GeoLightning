"""
Sensor Model
============

Sensor Model - Geolocation of Atmospheric Events

Summary
-------
This module defines the probabilistic detection model for atmospheric 
events by a distributed sensor network. It includes a function to compute 
the detection probability as a function of the distance between a sensor 
and an event, and a Monte Carlo-based simulation function to determine 
whether detection occurs.

The detection model is calibrated based on the StormDetector V2 device, 
and is suitable for simulating sparse and range-limited detection scenarios 
in lightning localization and similar geophysical applications.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Contents
--------
- get_detection_probability: computes probability of detection based on distance
- sensor_detection: probabilistically simulates whether an event is detected

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, 
Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True, fastmath=True)
def get_detection_probability(distance: np.float64) -> np.float64:
    """
    Compute the probability of detecting an atmospheric event 
    based on the distance to the sensor.

    This function models the detection likelihood based on a 
    linear decay of efficiency beyond a 120 km threshold. The 
    maximum probability is 96% and decreases linearly 
    as distance increases.

    Parameters
    ----------
    distance : np.float64
        Distance between the sensor and the event (in meters).

    Returns
    -------
    np.float64
        Detection probability (between 0 and 1).
    """
    if distance < 120_000.0:
        return 0.96
    else:
        p = (0.96 - 0.005 * (distance - 120_000.0) / 1_000.0)
        if p > 0.0:
            return p
        return 0.0


@jit(nopython=True, cache=True, fastmath=True)
def sensor_detection(distance: np.float64) -> bool:
    """
    Simulate the detection of an atmospheric event by a sensor 
    based on the detection probability.

    This function uses the `get_detection_probability` model 
    and performs a random draw from a uniform distribution 
    to decide if the event is detected.

    Parameters
    ----------
    distance : np.float64
        Distance between the sensor and the event (in meters).

    Returns
    -------
    bool
        True if the event was detected; False otherwise.
    """
    probability = get_detection_probability(distance)
    rand_int = np.random.random_sample()
    return rand_int <= probability

if __name__ == "__main__":
    for i in range(100):
        distance = np.random.uniform(20_000.0, 200_000.0)
        print(distance, sensor_detection(distance))