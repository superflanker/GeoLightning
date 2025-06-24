"""
    EELT 7019 - Inteligência Artificial Aplicada
    Teste do Tuner com GWO
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from GeoLightning.HyperParameters.Tuning import tuneit
from mealpy.swarm_based.GWO import OriginalGWO

def test_tuneit_GWO():

    params_grid = {
        "epoch": [10, 100, 1000],
        "pop_size": [10, 100]
    }

    term = {

        "max_epoch": 2000,

    }

    best_dict = tuneit(OriginalGWO(), params_grid, term)
    print(best_dict)

if __name__ == "__main__":
    test_tuneit_GWO()
