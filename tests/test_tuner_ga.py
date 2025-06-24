"""
    EELT 7019 - Inteligência Artificial Aplicada
    Teste do Tuner com GA
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from GeoLightning.HyperParameters.Tuning import tuneit
from mealpy.evolutionary_based.GA import BaseGA

def test_tuneit_GA():

    params_grid = {
        "epoch": [10, 100],
        "pop_size": [10, 100],
        "pc": [0.01, 0.05, 0.1, 0.11, 0.13, 0.2],
        "pm": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    }

    term = {

        "max_epoch": 2000,

    }

    best_dict = tuneit(BaseGA(), params_grid, term)
    print(best_dict)

if __name__ == "__main__":
    test_tuneit_GA()
