"""
    EELT 7019 - Inteligência Artificial Aplicada
    Teste do Algoritmo Principal
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.Stela import stela

def test_stela_execucao_sintetica():
    solucoes = np.array([[0,0,0], [0.1,0,0], [5,5,5], [5.1,5,5]])
    tempos_chegada = np.array([1.0, 1.1, 5.0, 5.1])
    estacoes = np.array([[0,0,0], [0,1,0], [5,5,5], [5,6,5]])
    clusters = np.array([0,0,1,1])
    verossimilhanca = stela(solucoes, tempos_chegada, estacoes, clusters, sistema_cartesiano=True)
    assert isinstance(verossimilhanca, float)
