
"""
EELT 7019 - Applied Artificial Intelligence
Cluster Remapping Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.Dimensions import remapeia_solucoes, remapeia_solucoes_unicas

def test_remapeamento_com_cluster_e_ruido():
    solucoes = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],
                         [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0],
                         [3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0],
                         [4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0],
                         [4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]])
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2,
                      3, 3, 3, 4, 4, 4, 4, 4, 4])
    centroides = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0],
                           [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]])
    
    novas_solucoes = remapeia_solucoes(solucoes,
                                       labels,
                                       centroides)
    
    solucoes_unicas = remapeia_solucoes_unicas(labels)
    
    assert len(novas_solucoes) == len(solucoes)
    assert len(np.unique(solucoes_unicas)) == 2
