"""
    EELT 7019 - Inteligência Artificial Aplicada
    Teste do Tuner com AOA
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from GeoLightning.HyperParameters.Tuning import tuneit
from mealpy.math_based.AOA import OriginalAOA

def test_tuneit_AOA():

    params_grid = {
        "epoch": [10],
        "pop_size": [10],
        "alpha": [3, 4, 5, 6, 7, 8],
        "miu": [0.5, 0.7, 0.8, 1.0],
        "moa_min": [0.1, 0.2, 0.3, 0.4],
        "moa_max": [0.5, 0.8, 1.0]
    }

    term = {

        "max_epoch": 2000,

    }

    best_dict = tuneit(OriginalAOA(), params_grid, term)
    print(best_dict)

if __name__ == "__main__":
    test_tuneit_AOA()
