
import numpy as np
from GeoLightning.Stela.Dimensions import remapeia_solucoes

def test_remapeamento_com_cluster_e_ruido():
    centroides = np.array([[0.0, 0.0, 0.0]])
    solucoes = np.array([[0.0, 0.0, 0.0],
                         [1.0, 1.0, 1.0],
                         [2.0, 2.0, 2.0]])
    labels = np.array([0, -1, -1])
    novas_solucoes, solucoes_unicas = remapeia_solucoes(solucoes, labels, centroides)
    assert novas_solucoes.shape[0] == 3
    assert solucoes_unicas.shape[0] == 3
