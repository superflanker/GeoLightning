"""
EELT 7019 - Applied Artificial Intelligence
Tuner Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""


from GeoLightning.HyperParameters.Tuning import tuneit
from mealpy.swarm_based.GWO import OriginalGWO

def test_tuneit_GWO(fake_test=False):

    params_grid = {
        "epoch": [10],
        "pop_size": [10]
    }

    term = {
        "max_epoch": 200,
    }

    if fake_test:

        params_grid = {
            "epoch": [10, 100, 1000],
            "pop_size": [10, 100]
        }

        term = {
            "max_epoch": 2000,
        }

    try:
        if fake_test:    
            best_dict = tuneit(OriginalGWO(), params_grid, term)
            print(best_dict)
    except:
        pass

if __name__ == "__main__":
    test_tuneit_GWO(True)
