Global System Constants
=============================

This page presents the fundamental constants used in the geolocation modules of the **GeoLightning** project, with emphasis on physical, temporal, spatial and clustering parameters.

Physical Constants
------------------

.. py:data:: AVG_EARTH_RADIUS

float: Mean radius of the Earth in meters (6371.0088 km).

.. py:data:: AVG_LIGHT_SPEED

float: Speed ​​of light in vacuum, in meters per second.

Statistical Constants
-----------------------

.. py:data:: SIGMA_T

float: Standard deviation of temporal uncertainty (in seconds).

.. py:data:: SIGMA_D

float: Equivalent spatial standard deviation (in meters), obtained by multiplying the speed of light by SIGMA_T.

Detection tolerances
-----------------------

.. py:data:: EPSILON_T

float: Maximum admissible temporal tolerance (1000 × SIGMA_T).

.. py:data:: EPSILON_D

float: Maximum admissible spatial tolerance (1000 × SIGMA_D).

.. py:data:: LIMIT_D

float: Spatial limit (in meters) for the global search for solutions by metaheuristic methods.

Clustering and range parameters
-----------------------------------

.. py:data:: CLUSTER_MIN_PTS

int: Minimum number of points to consider a valid cluster (DBSCAN).

.. py:data:: R_LAT

float: Conversion factor: meters per degree of latitude (approximately 111,320 m/°).

.. py:data:: MAX_DISTANCE

float: Maximum detection range of a sensor, in meters (160 km).