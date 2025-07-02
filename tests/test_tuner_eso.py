"""
EELT 7019 - Applied Artificial Intelligence
Tuner Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""


from GeoLightning.HyperParameters.Tuning import tuneit
from GeoLightning.Solvers.Mealpy.ESO import ESO


def test_tuneit_ESO(fake_test=False):

    params_grid = {
        "epoch": [10],
        "pop_size": [10]
    }

    term = {
        "max_epoch": 200,
    }

    if fake_test:

        params_grid = {
            "epoch": [10, 100, 200, 300, 400, 500],
            "pop_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        }

        term = {
            "max_epoch": 2000,
        }
    try:
        if fake_test:
            best_dict = tuneit(ESO(), params_grid, term)
            print(best_dict)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    test_tuneit_ESO(True)
