"""
EELT 7019 - Applied Artificial Intelligence
Tuner Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""


from GeoLightning.HyperParameters.Tuning import tuneit
from mealpy.swarm_based.PSO import OriginalPSO

def test_tuneit_PSO(fake_test=False):

    params_grid = {
        "epoch": [10],
        "pop_size": [10],
        "c1": [1.0],
        "c2": [1.0],
        "w": [0.5]
    }

    term = {
        "max_epoch": 200,
    }
    if fake_test:

        params_grid = {
            "epoch": [10, 100],
            "pop_size": [10, 100],
            "c1": [1.0, 1.5, 1.8, 2.0],
            "c2": [1.0, 1.5, 1.8, 2.0],
            "w": [0.5, 0.6, 0.7, 0.8]
        }

        term = {
            "max_epoch": 2000,
        }
    try:
        if fake_test:    
            best_dict = tuneit(OriginalPSO(), params_grid, term)
            print(best_dict)
    except:
        pass
if __name__ == "__main__":
    test_tuneit_PSO(True)
