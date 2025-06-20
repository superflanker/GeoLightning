"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes DBSCAN3D
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.DBSCAN3D import clusterizacao_DBSCAN3D

def test_dbscan3d():
    cluster1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64)
    cluster2 = np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]], dtype=np.float64)
    cluster3 = np.array([[6, 6, 6], [6, 6, 6], [6, 6, 6]], dtype=np.float64)
    solucoes = np.vstack((cluster1, cluster2, cluster3))
    labels = clusterizacao_DBSCAN3D(solucoes,
                                    eps=1.0,
                                    min_pts=3,
                                    sistema_cartesiano=True)
    
    assert len(np.unique(labels[labels >= 0])) == 3
