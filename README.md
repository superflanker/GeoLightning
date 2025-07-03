# GeoLightning: Detection and Localization of Lightning Discharges Using STELA

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

GeoLightning is a Python package for the **spatiotemporal estimation of geophysical events**, with emphasis on **lightning discharge localization** using the **STELA algorithm**. It supports advanced **metaheuristic optimization** through seamless integration with the [MEALPY](https://pypi.org/project/mealpy/) framework.

---

## 🔍 Overview

This project models the detection of geophysical events by multiple distributed sensors. The key algorithm, `stela()`, performs **spatial and temporal clustering** based on time-of-arrival (TOA) data and location information.

GeoLightning supports flexible optimization by wrapping the problem in a `StelaProblem` class compatible with any MEALPY algorithm. A specialized version `StelaPSO` and others are just convenience wrappers for our research.

---

## 📦 Main Components

- `StelaProblem`: Problem wrapper for MEALPY optimizers.
- `StelaPSO`: Custom PSO convenience wrapper.
- `StelaAOA`: Custom AOA convenience wrapper.
- `StelaGWO`: Custom GWO convenience wrapper.
- `StelaGA`: Custom GA convenience wrapper.
- `StelaLSA`: Custom LSA convenience wrapper.
- `StelaFHO`: Custom FHO convenience wrapper.
- `StelaESO`: Custom ESO convenience wrapper.
- `stela_phase_one()`, `stela_phase_two()`: Core algorithm performing spatiotemporal clustering and likelihood estimation.

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
│   ├── GraphUtils/            # Plot Utilities
│   ├── HyperParameters/       # Parameter Tuning Utility
│   ├── Runners/               # Convenience Wrappers for Mealpy Solvers
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
problem = StelaProblem(bounds, "min", arrival_points, arrival_times)
problem.cluster_it()
model = StelaPSO(epoch=100, pop_size=50)
best_solution, best_fitness = model.solve(problem)
```
---

## 📄 Reference

This implementation is based on the following research article:

> **Adams, A.M.**, *Origin Estimation and Event Separation in Time-of-Arrival Localization of Lightning Events*, 2025 (in review).

---

## 📄 Article References

1. A. Nag, M. J. Murphy, W. Schulz, and K. L. Cummins, “Lightning locating systems: Insights on characteristics and validation techniques,” Earth and Space Science, vol. 2, pp. 65–93, 2015. [Link](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2014EA000051)

2. K. L. Cummins and M. J. Murphy, “An overview of lightning locating systems: History, techniques, and data uses, with an in-depth look at the us nldn,” in IEEE, 2009. [Link](https://ieeexplore.ieee.org/abstract/document/5173582/)

3. F. Ahmed, S. Hasan, I. M. Mahbubul, and M. A. K. Mallik, “Gis-based spatial analysis for lightning scenario in bangladesh,” Heliyon, 2024. [Link](https://www.cell.com/heliyon/fulltext/S2405-8440(24)04739-X)

4. H. Ali, M. M. Aman, and S. J. Ahmed, “Lightning flash density map of pakistan on arc-gis software-anempirical approach,” in IEEE, 2018. [Link](https://ieeexplore.ieee.org/abstract/document/8494579/)

5. S. Pédeboy, “An introduction to the iec 62858: lightning density based on lightning locating systems,” International Lightning Protection Synposium, Shenzen, China, October, vol. 2018, 2018.

6. O. P. Jr, I. Pinto, and K. P. Naccarato, “Maximum cloud-to-ground lightning flash densities observed by lightning location systems in the tropical region: A review,” Atmospheric Research, 2007. [Link](https://www.sciencedirect.com/science/article/pii/S0169809506002845)

7. K. Mehranzamir, A. B. Pour, Z. Abdul-Malek, and H. N. Afrouzi, “Implementation of ground-based lightning locating system using particle swarm optimization algorithm for lightning mapping and monitoring,” Remote Sensing, 2023. [Link](https://www.mdpi.com/2072-4292/15/9/2306)

8. J. Rajendran, K. H. Minu, and K. Mohan, “Advances in lightning electromagnetic field sensors and geolocation techniques: A review,” in IEEE, 2025. [Link](https://ieeexplore.ieee.org/abstract/document/10988202/)

9. B. Salimi, “Time difference of arrival-based three-station lighting locating system in malaysia,” Ph.D. dissertation, Universiti Teknologi Malaysia, 2015. [Link](http://eprints.utm.my/54808/1/BehnamSalimiPFKE2015.pdf)

10. Y. Zhang, Y. Zhang, M. Zou, J. Wang, Y. Li, Y. Tan, and Y. Feng, “Advances in lightning monitoring and location technology research in china,” Remote Sensing, 2022. [Link](https://www.mdpi.com/2072-4292/14/5/1293)

11. S. Dutta, N. S. Jayalakshmi, and V. K. Jadoun, “Shifting of research trends in fault detection and estimation of location in power system,” in IEEE, 2025. [Link](https://ieeexplore.ieee.org/abstract/document/10969631/)

12. M. Adnane, A. Khoumsi, and J. P. F. Trovão, “Efficient management of energy consumption of electric vehicles using machine learning—a systematic and comprehensive survey,” Energies, 2023. [Link](https://www.mdpi.com/1996-1073/16/13/4897)

13. T. S. Delwar, U. Aras, S. Mukhopadhyay, and A. Kumar, “The intersection of machine learning and wireless sensor network security for cyber-attack detection: a detailed analysis,” Sensors, 2024. [Link](https://www.mdpi.com/1424-8220/24/19/6377)

14. A. L. Fata, I. Tosi, and M. Brignone, “A review of lightning location systems: part i-methodologies and techniques,” in IEEE, 2020. [Link](https://ieeexplore.ieee.org/abstract/document/9160534/)

15. A. Alammari, A. A. Alkahtani, and M. R. Ahmad, “Lightning mapping: Techniques, challenges, and opportunities,” in IEEE, 2020. [Link](https://ieeexplore.ieee.org/abstract/document/9226420/)

16. I. Sharp and K. Yu, Wireless Positioning: Principles and Practice. Springer, 2019. [Link](https://link.springer.com/content/pdf/10.1007/978-981-10-8791-2.pdf)

17. H. Shen, Z. Ding, S. Dasgupta, and C. Zhao, “Multiple source localization in wireless sensor networks based on time of arrival measurement,” IEEE Transactions on Signal Processing, vol. 62, no. 8, pp. 1938–1949, 2014.

18. H. Jamali-Rad and G. Leus, “Sparsity-aware multi-source tdoa localization,” IEEE Transactions on Signal Processing, vol. 61, no. 19, pp. 4874–4887, 2013.

19. X. Guo, Z. Chen, X. Hu, and X. Li, “Multi-source localization using time of arrival self-clustering method in wireless sensor networks,” IEEE Access, vol. 7, pp. 82 110–82 121, 2019.

20. K. C. Ho and T.-K. Le, “Integrating aoa with tdoa for joint source and sensor localization,” IEEE Transactions on Signal Processing, vol. 71, pp. 2087–2102, 2023.

21. M. Delcourt and J.-Y. Le Boudec, “Tdoa source-localization technique robust to time-synchronization attacks,” IEEE Transactions on Information Forensics and Security, vol. 16, pp. 4249–4264, 2021.

22. Y. Han, B. He, and H. Shu, “Location distribution of lightning localization monitoring stations integrating 3d monitoring and pso algorithm,” Processes, 2024. [Link](https://www.mdpi.com/2227-9717/13/1/2)

23. J. W. Tang, V. Cooray, and C. L. Wooi, “Optimization of lightning protection system using multi-objective optimization techniques,” in IEEE, 2024. [Link](https://ieeexplore.ieee.org/abstract/document/10832672/)

24. A. Alammari, A. A. Alkahtani, M. R. Ahmad, and A. Aljanad, “Cross-correlation wavelet-domain-based particle swarm optimization for lightning mapping,” Applied Sciences, 2021. [Link](https://www.mdpi.com/2076-3417/11/18/8634)

25. S. Deb, X. Z. Gao, K. Tammi, and K. Kalita, “Nature-inspired optimization algorithms applied for solving charging station placement problem: overview and comparison,” Archives of Computational Methods in Engineering, 2021. [Link](https://link.springer.com/article/10.1007/s11831-019-09374-4)

26. L. Abualigah, M. A. Elaziz, A. G. Hussien, and B. Alsalibi, “Lightning search algorithm: a comprehensive survey,” Applied Intelligence, 2021. [Link](https://link.springer.com/article/10.1007/s10489-020-01947-2)

27. M. Azizi, S. Talatahari, and A. H. Gandomi, “Fire hawk optimizer: a novel metaheuristic algorithm,” Artificial Intelligence Review, vol. 56, pp. 287–363, 2023. [Link](https://doi.org/10.1007/s10462-022-10173-w)

28. M. Soto Calvo and H. S. Lee, “Electrical storm optimization (eso) algorithm: Theoretical foundations, analysis, and application to engineering problems,” Machine Learning and Knowledge Extraction, vol. 7, no. 1, 2025. [Link](https://www.mdpi.com/2504-4990/7/1/24)

29. Z. Hu, Y. Wen, W. Zhao, and H. Zhu, “Particle swarm optimization-based algorithm for lightning location estimation,” in 2010 Sixth International Conference on Natural Computation, vol. 5, 2010, pp. 2668–2672.

30. Y. Chan, H. Hang, and P. Ching, “Exact and approximate maximum likelihood localization algorithms,” in 2006 IEEE International Conference on Acoustics Speech and Signal Processing Proceedings, vol. 4. IEEE, 2006, pp. IV–IV. [Link](https://ieeexplore.ieee.org/abstract/document/1583909/)

31. R. Rawassizadeh, C. Dobbins, M. Akbari, and M. Pazzani, “Indexing multivariate mobile data through spatio-temporal event detectionand clustering,” Sensors, vol. 19, no. 3, 2019. [Link](https://www.mdpi.com/1424-8220/19/3/448)

32. F. Elvander, I. Haasler, A. Jakobsson, and J. Karlsson, “Multi-marginal optimal transport using partial information with applications in robust localization and sensor fusion,” Signal Processing, vol. 171, p. 107474, 2020. [Link](https://www.sciencedirect.com/science/article/pii/S0165168420300207)

33. C. M. Bishop, Pattern Recognition and Machine Learning. New York: Springer, 2006.

34. E. Xu, Z. Ding, and S. Dasgupta, “Source localization in wireless sensor networks from signal time-of-arrival measurements,” IEEE Transactions on Signal Processing, vol. 59, no. 6, pp. 2795–2805, 2011. [Link](https://ieeexplore.ieee.org/abstract/document/5714759/)

35. J. Salt, H. Nguyen, and N. Pham, “Probabilistic source localization based on time-of-arrival measurements,” in 2020 International Conference on Communications (ICC). IEEE, 2020, pp. 1–6. [Link](https://ieeexplore.ieee.org/abstract/document/9241833/)

36. X.-S. Yang, Engineering Optimization: An Introduction with Metaheuristic Applications. Hoboken, NJ: John Wiley & Sons, 2010.

37. K. Mehranzamir, A. B. Pour, and Z. A. M. Abdul-Malek, “Particle swarm optimization (pso) algorithm for ground-based lightning locating system (glls) in johor, malaysia: Hazard mitigation implications,” 2022. [Link](https://www.researchsquare.com/article/rs-1711210/latest)

38. O. O. Obadina, M. A. Thaha, K. Althoefer, and M. H. Shaheed, “Dynamic characterization of a master–slave robotic manipulator using a hybrid grey wolf–whale optimization algorithm,” Journal of Vibration and Control, vol. 28, no. 15-16, pp. 1992–2003, 2022. [Link](https://doi.org/10.1177/10775463211003402)

39. L. Abualigah, A. Diabat, S. Mirjalili, M. Abd Elaziz, and A. H. Gandomi, “The arithmetic optimization algorithm,” Computer Methods in Applied Mechanics and Engineering, vol. 376, p. 113609, 2021. [Link](https://www.sciencedirect.com/science/article/pii/S0045782520307945)

40. Python Core Team, Python: A dynamic, open source programming language, Python Software Foundation, 2024. [Link](https://www.python.org/)

41. A. M. Adams, “Geolightning: Detection and localization of lightning discharges using stela,” https://github.com/superflanker/GeoLightning, 2025, accessed: July 3, 2025.

42. M. Geyer, “Aircraft navigation and surveillance analysis for a spherical earth,” John A. Volpe National Transportation Systems Center (U.S.), Technical Report DOT-VNTSC-FAA-15-01, October 2014, prepared for the United States Federal Aviation Administration, Wake Turbulence Research Office. [Link](https://rosap.ntl.bts.gov/view/dot/12122)

43. D. Deng, “Dbscan clustering algorithm based on density,” in 2020 7th International Forum on Electrical Engineering and Automation (IFEEA), 2020, pp. 949–953.

44. W. Lin, J. Wang, B. Ren, J. Yu, X. Wang, and T. Zhang, “Robust optimization of rolling parameters of coarse aggregates based on improved response surface method using satisfaction function method based on entropy and adaptive chaotic gray wolf optimization,” Construction and Building Materials, vol. 316, p. 125839, 2022. [Link](https://www.sciencedirect.com/science/article/pii/S0950061821035728)

45. C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau, E. Wieser, J. Taylor, S. Berg, N. J. Smith, R. Kern, M. Picus, S. Hoyer, M. H. van Kerkwijk, M. Brett, A. Haldane, J. Fernández del Rı́o, M. Wiebe, P. Peterson, P. Gérard-Marchant, K. Sheppard, T. Reddy, W. Weckesser, H. Abbasi, C. Gohlke, andT. E. Oliphant, “Array programming with NumPy,” Nature, vol. 585, p. 357–362, 2020.

46. S. K. Lam, A. Pitrou, and S. Seibert, “Numba: A llvm-based python jit compiler,” in Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC, 2015, pp. 1–6.

47. N. Van Thieu and S. Mirjalili, “Mealpy: An open-source library for latest meta-heuristic algorithms in python,” Journal of Systems Architecture, 2023.

48. H. Liu, Y. Chen, Y. Huang, X. Cheng, and Q. Xiao, “Study on the localization method of multi-aperture acoustic array based on tdoa,” IEEE Sensors Journal, vol. 21, no. 12, pp. 13 805–13 814, 2021.

## 📚 How to Cite

If you use **GeoLightning** in your research, please cite:

```bibtex
@misc{adams2025geolightning,
  author    = {Augusto Mathias Adams},
  title     = {GeoLightning: Detection and Localization of Lightning Discharges Using STELA},
  year      = {2025},
  howpublished = {\url{https://github.com/superflanker/GeoLightning}}
}
```
