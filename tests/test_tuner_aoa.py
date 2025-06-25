"""
EELT 7019 - Applied Artificial Intelligence
Tuner Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from GeoLightning.HyperParameters.Tuning import tuneit
from mealpy.math_based.AOA import OriginalAOA


def test_tuneit_AOA(fake_test=False):

    params_grid = {
        "epoch": [10],
        "pop_size": [10],
        "alpha": [3],
        "miu": [0.5],
        "moa_min": [0.1],
        "moa_max": [0.5]
    }

    term = {
        "max_epoch": 20,
    }

    if fake_test:

        params_grid = {
            "epoch": [10, 50, 100, 150, 200],
            "pop_size": [10, 50, 100],
            "alpha": [3, 4, 5],
            "miu": [0.5],
            "moa_min": [0.1],
            "moa_max": [0.5]
        }

        term = {
            "max_epoch": 2000,
        }
    try:
        if fake_test:
            best_dict = tuneit(OriginalAOA(), params_grid, term)
            print(best_dict)
    except:
        pass


if __name__ == "__main__":
    test_tuneit_AOA(True)
