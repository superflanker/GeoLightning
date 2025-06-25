"""
EELT 7019 - Applied Artificial Intelligence
Tuner Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""


from GeoLightning.HyperParameters.Tuning import tuneit
from GeoLightning.Solvers.Mealpy.LSA import LSA


def test_tuneit_LSA(fake_test=False):

    params_grid = {
        "epoch": [10],
        "pop_size": [10]
    }

    term = {
        "max_epoch": 200,
    }

    if fake_test:
        params_grid = {
            "epoch": [10, 100, 200, 1000, 1500, 2000],
            "pop_size": [10, 50, 100]
        }

        term = {
            "max_epoch": 2000,
        }

    try:
        if fake_test:
            best_dict = tuneit(LSA(), params_grid, term)
            print(best_dict)
    except:
        pass


if __name__ == "__main__":
    test_tuneit_LSA(True)
