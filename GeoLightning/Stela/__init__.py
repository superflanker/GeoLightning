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
- DBSCAN.py: DBSCAN Algorithm using a true spatio-temporal metric (deprecated)
- IRLS.py: robust position refinement through IRLS Algorithm + LM update
- LogLikelihood.py: Log likelihood computation.
- PivotClustering.py: TDOA Based Phisically Informed 1D Pivoting Clustering Algorithm for event separation.
- Stela.py: Core spatio-temporal likelihood computation.

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
"""
