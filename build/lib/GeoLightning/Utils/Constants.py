"""
Constants
=========

Constants for Atmospheric Event Geolocation
-------------------------------------------

This module defines global constants used in the simulation and estimation 
of atmospheric event locations. It includes physical constants, 
sensor-related noise parameters, spatial tolerances, and thresholds 
for spatiotemporal clustering and localization bounds.

Author
-------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Summary
--------
The constants defined in this module are used across multiple components
of the STELA pipeline and simulation tools. These include:
- Physical constants such as the average Earth radius and speed of light.
- Noise characteristics of detection systems (e.g., standard deviations).
- Spatial and temporal tolerances for clustering and validation.
- Thresholds for determining bounds and sensor coverage.

Notes
-----
This module is part of the activities of the discipline 
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
numpy
"""

import numpy as np

AVG_EARTH_RADIUS: np.float64 = 1000 * 6371.0088
"""float: Mean radius of the Earth in meters."""

AVG_LIGHT_SPEED: np.float64 = 299_792_458.0
"""float: Speed of light in vacuum, in meters per second."""

SIGMA_T: np.float64 = 1.0e-6
"""float: Temporal standard deviation of detection uncertainty (in seconds)."""

SIGMA_D: np.float64 = AVG_LIGHT_SPEED * SIGMA_T
"""float: Spatial standard deviation corresponding to SIGMA_T (in meters)."""

EPSILON_T: np.float64 = 1000 * SIGMA_T
"""float: Maximum admissible temporal tolerance (in seconds)."""

EPSILON_D: np.float64 = 1000 * SIGMA_D
"""float: Maximum admissible spatial tolerance (in meters)."""

LIMIT_D: np.float64 = 10 * SIGMA_D
"""float: Maximum search radius for metaheuristic methods (in meters)."""

CLUSTER_MIN_PTS: np.int32 = 3
"""int: Minimum number of points to form a valid cluster."""

R_LAT: np.float64 = 111_320.0
"""float: Latitude conversion factor — meters per degree."""

MAX_DISTANCE: np.float64 = 160_000.0
"""float: Range maximum detection of a sensor (in meters)."""