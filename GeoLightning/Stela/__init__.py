"""
STELA Core Module
-----------------

Summary
-------
Contains the core implementation of the STELA algorithm 
(Spatio-Temporal Likelihood Estimation for Lightning Analysis), 
used to identify and separate multiple atmospheric events based on 
TOA (Time-of-Arrival) data.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Submodules
----------
- Bounds.py: spatio-tempral bounds computation.
- Common.py: common utilities for the module.
- DBSCAN3D.py: DBSCAN 3D optimized with numba.
- Dimensions.py: Remapping of Candidate Solutions.
- Entropy.py: Shannon Entropy computation.
- LogLikelihood.py: Log likelihood computation.
- SpatialClustering.py: Spatial Clustering and Log-likelihood computation.
- STDBSCAN.py: Spatio-Temporal DBSCAN (used for comparisions)
- Stela.py: Core spatio-temporal likelihood computation.
- TemporalClustering.py: Temporal Clustering using 1D DBSCAN.

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
"""
