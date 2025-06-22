"""
    EELT 7019 - Inteligência Artificial Aplicada
    Teste de Clusterização Temporal
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.Deprecated.TemporalClustering import clusterizacao_temporal_stela

def test_temporal_clustering_dois_grupos():
    tempos = np.array([1.0, 1.001, 1.002, 5.0, 5.001, 5.002])
    labels, medias, detectores = clusterizacao_temporal_stela(tempos, eps=0.01, min_pts=2)
    assert len(np.unique(labels[labels >= 0])) == 2
    assert medias.shape[0] == 2
