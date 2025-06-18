"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes de Clusteruzação Espacial
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.SpatialClustering import clusterizacao_espacial_stela

def test_clusterizacao_espacial_stela_sintetico():
    solucoes = np.array([[0,0,0], [0,0.1,0], [5,5,5], [5.1,5,5]])
    clusters_temporais = np.array([0,0,1,1])
    tempos = np.array([1.0, 1.1, 5.0, 5.1])
    tempos_medios = np.array([1.05, 5.05])
    novas_solucoes, loglik = clusterizacao_espacial_stela(solucoes, clusters_temporais, tempos, tempos_medios, eps=0.5, sigma_t=1.0, sigma_d=1.0, min_pts=2, sistema_cartesiano=True)
    assert novas_solucoes.shape[0] > 0
    assert isinstance(loglik, float)
