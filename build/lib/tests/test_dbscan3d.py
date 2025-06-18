"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes DBSCAN3D
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.DBSCAN3D import clusterizacao_DBSCAN3D

def test_dbscan3d_com_dois_clusters_distintos():
    cluster1 = np.array([[0,0,0], [0,0.1,0], [0,0.2,0]])
    cluster2 = np.array([[5,5,5], [5,5.1,5], [5,5.2,5]])
    cluster3 = np.array([[6,8,8], [6,8.1,8], [6,8.2,8]])
    solucoes = np.vstack((cluster1, cluster2, cluster3))
    labels, centroides, distancias, detectores = clusterizacao_DBSCAN3D(solucoes, 
                                                                        eps=0.5, 
                                                                        min_pts=3, 
                                                                        sistema_cartesiano=True)
    assert len(np.unique(labels[labels >= 0])) == 2
    assert centroides.shape[0] == 2

if __name__ == "__main__":
    test_dbscan3d_com_dois_clusters_distintos()