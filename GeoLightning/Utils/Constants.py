"""
EELT 7019 - Applied Artificial Intelligence
===========================================

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

# Average radius of Earth (in meters)
AVG_EARTH_RADIUS: np.float64 = 1000 * 6371.0088

# Average speed of light in vacuum (in meters per second)
AVG_LIGHT_SPEED: np.float64 = 299_792_458.0

# Temporal standard deviation (sensor uncertainty, in seconds)
SIGMA_T: np.float64 = 1.0e-6

# Spatial standard deviation, derived from temporal uncertainty
SIGMA_D: np.float64 = AVG_LIGHT_SPEED * SIGMA_T

# Maximum admissible deviations
# Temporal tolerance
EPSILON_T: np.float64 = 1000 * SIGMA_T
# Spatial tolerance
EPSILON_D: np.float64 = 1000 * SIGMA_D

# Maximum bounding radius for metaheuristic search (in meters)
LIMIT_D: np.float64 = 10 * SIGMA_D

# Minimum number of points required to form a valid cluster
CLUSTER_MIN_PTS: np.int32 = 3

# Conversion factor: meters per degree of latitude
R_LAT: np.float64 = 111_320.0

# Maximum detection range of a sensor (in meters)
MAX_DISTANCE: np.float64 = 160_000.0

