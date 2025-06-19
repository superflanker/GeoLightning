"""
    EELT 7019 - Inteligência Artificial Aplicada  
    Wrapper FHO do STELA  
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>  
"""

import numpy as np
from GeoLightning.Solvers.Mealpy.FHO import FHO
from GeoLightning.Solvers.StelaProblem import StelaProblem


class StelaFHO(FHO):
    """
    Classe especializada que estende o otimizador FHO da MEALPY
    para operar diretamente sobre instâncias da classe `StelaProblem`.

    Antes de cada iteração evolutiva, o espaço de busca é adaptativamente 
    refinado com base na melhor solução encontrada até o momento, por meio
    do método `restart_search_space()` do problema STELA.
    """

    def evolve(self, pop=None):
        """
        Executa uma iteração do algoritmo FHO com refinamento adaptativo
        do espaço de busca.

        Este método substitui a versão padrão do MEALPY, integrando o 
        mecanismo de atualização de limites definido na classe `StelaProblem`.

        Args:
            pop (list, optional): População atual de agentes (partículas). 
                                  Caso não fornecida, utiliza a população interna.

        Raises:
            TypeError: Caso o problema associado não seja uma instância de `StelaProblem`.
        """
        if not isinstance(self.problem, StelaProblem):
            raise TypeError("O problema fornecido deve ser uma instância de StelaProblem.")
        self.problem.restart_search_space()
        super().evolve(pop)
