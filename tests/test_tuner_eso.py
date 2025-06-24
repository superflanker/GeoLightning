"""
    EELT 7019 - Inteligência Artificial Aplicada
    Teste do Tuner com ESO
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from GeoLightning.HyperParameters.Tuning import tuneit
from GeoLightning.Solvers.Mealpy.ESO import ESO

def test_tuneit_ESO():

    params_grid = {
        "epoch": [10, 100, 200, 1000],
        "pop_size": [10, 100]
    }

    term = {

        "max_epoch": 2000,

    }

    best_dict = tuneit(ESO(), params_grid, term)
    print(best_dict)

if __name__ == "__main__":
    test_tuneit_ESO()
