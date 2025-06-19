# GeoLightning: Detection and Localization of Lightning Discharges Using STELA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GeoLightning is a Python package for the **spatiotemporal estimation of geophysical events**, with emphasis on **lightning discharge localization** using the **STELA algorithm**. It supports advanced **metaheuristic optimization** through seamless integration with the [MEALPY](https://pypi.org/project/mealpy/) framework.

---

## 🔍 Overview

This project models the detection of geophysical events by multiple distributed sensors. The key algorithm, `stela()`, performs **spatial and temporal clustering** based on time-of-arrival (TOA) data and location information.

GeoLightning supports flexible optimization by wrapping the problem in a `StelaProblem` class compatible with any MEALPY algorithm. A specialized version `StelaPSO` (and others) adds adaptive search space refinement to improve convergence.

---

## 📦 Main Components

- `StelaProblem`: Problem wrapper for MEALPY optimizers.
- `StelaPSO`: Custom PSO with search-space refinement.
- `StelaAOA`: Custom AOA with search-space refinement.
- `StelaGWO`: Custom GWO with search-space refinement.
- `StelaGA`: Custom GA with search-space refinement.
- `StelaLSA`: Custom LSA with search-space refinement.
- `StelaFHO`: Custom FHO with search-space refinement.
- `StelaESO`: Custom ESO with search-space refinement.
- `stela()`: Core algorithm performing spatiotemporal clustering and likelihood estimation.

---

## 🛠 Dependencies

GeoLightning relies on the following Python libraries:

- [NumPy](https://pypi.org/project/numpy/)
- [SciPy](https://pypi.org/project/scipy/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Pandas](https://pypi.org/project/pandas/)
- [Numba](https://pypi.org/project/numba/)
- [MEALPY](https://pypi.org/project/mealpy/)
- [opfunu](https://pypi.org/project/opfunu/)
- [pytest](https://pypi.org/project/pytest/)
- [pytest-cov](https://pypi.org/project/pytest-cov/)
- [requests](https://pypi.org/project/requests/)
- [python-dateutil](https://pypi.org/project/python-dateutil/)
- [typing-extensions](https://pypi.org/project/typing-extensions/)
- [urllib3](https://pypi.org/project/urllib3/)
- [six](https://pypi.org/project/six/)
- [tzdata](https://pypi.org/project/tzdata/)
- and others (see `setup.py` or `requirements.txt` for full list)

---

## 📁 Project Structure

```
GeoLightning/
├── GeoLightning/
│   ├── Simulator/             # Sensor modeling and synthetic data
│   ├── Solvers/               # STELA wrappers and metaheuristics
│   ├── Stela/                 # Core spatiotemporal clustering and likelihood
│   └── Utils/                 # Coordinate tools and constants
├── tests/                     # Unit tests for each core module
├── LICENSE
├── README.md
└── setup.py
```

---

## 🧪 Running Experiments

To run an optimization experiment with any MEALPY algorithm:

```python
from GeoLightning.Solvers.StelaPSO import StelaPSO
from GeoLightning.Solvers.StelaProblem import StelaProblem

# Define bounds, arrival points and times...
problem = StelaProblem(bounds, "max", arrival_points, arrival_times)
model = StelaPSO(problem, epoch=100, pop_size=50)
best_solution, best_fitness = model.solve()
```

---

## 📄 Reference

This implementation is based on the following research article:

> **Adams, A.M.**, *Origin Estimation and Event Separation in Time-of-Arrival Localization of Lightning Events*, 2025 (in review).

---

## 📚 How to Cite

If you use **GeoLightning** in your research, please cite:

```bibtex
@misc{adams2025geolightning,
  author    = {Augusto Mathias Adams},
  title     = {GeoLightning: Detection and Localization of Lightning Discharges Using STELA},
  year      = {2025},
  howpublished = {\url{https://github.com/superflaker/GeoLightning}}
}
```
