"""
EELT 7019 - Applied Artificial Intelligence
Tuner Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""


from GeoLightning.HyperParameters.Tuning import tuneit
from mealpy.evolutionary_based.GA import BaseGA


def test_tuneit_GA(fake_test=False):
    params_grid = {
        "epoch": [10],
        "pop_size": [10],
        "pc": [0.01],
        "pm": [0.7]
    }

    term = {
        "max_epoch": 200,
    }
    if fake_test:
        params_grid = {
            "epoch": [10, 100, 200, 300, 400, 500],
            "pop_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "pc": [0.01, 0.05, 0.1, 0.11, 0.13, 0.2],
            "pm": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        }

        term = {
            "max_epoch": 2000,
        }

    try:
        if fake_test:
            best_dict = tuneit(BaseGA(), params_grid, term)
            print(best_dict)
    except:
        pass


if __name__ == "__main__":
    test_tuneit_GA(True)
