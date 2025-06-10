"""
    EELT 7019 - Inteligência Artificial Aplicada
    Algoritmo STELA - Clusterização Espacial baseada na hierarquia temporal
    (2. fase)
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
from numba import jit
import numpy as np
from GeoLightning.Utils.Constants import SIGMA_T, \
    SIGMA_D, \
    EPSILON_D, \
    CLUSTER_MIN_PTS
from GeoLightning.Utils.Utils import concat_manual
from .DBSCAN3D import clusterizacao_DBSCAN3D
from .LogLikelihood import funcao_log_verossimilhanca
from .Dimensions import remapeia_solucoes

@jit(nopython=True, cache=True, fastmath=True)
def clusterizacao_espacial_stela(solucoes: np.ndarray,
                                 clusters: np.ndarray,
                                 tempos: np.ndarray,
                                 tempos_medios: np.ndarray,
                                 eps: np.float64 = EPSILON_D,
                                 sigma_t: np.float64 = SIGMA_T,
                                 sigma_d: np.float64 = SIGMA_D,
                                 min_pts: np.int32 = CLUSTER_MIN_PTS,
                                 sistema_cartesiano: bool = False) -> tuple:
    """
    Clusterização espacial do STELA - determinação de centroides e agrupamento
    Args: 
        solucoes (np.ndarray): array (N, 3) de posições candidatas
        clusters (np.ndarray): vetor com os rótulos de cluster 
                               atribuídos a cada Solução
        tempos (np.ndarray): vetor com os tempos de origem estimados
        tempos_medios (np.ndarray): os tempos médios de cada cluster, 
                                    para cálculo da função de ajuste
        detectores_por_cluster (np.ndarray): detectores envolvidos em um cluster
        eps (np.float64): tolerância máxima (janela espacial) 
                          para definição de vizinhança
        sigma_t (np.float64): desvio padrão temporal
        sigma_d (np.float64): desvio padrão espacial
        min_pts (np.int32): número mínimo de pontos para formar um cluster
        sistema_cartesiano (bool): usa distância euclidiana (True) 
                ou esférica (False)
    Returns:
        tuple =>    solucoes_unicas (np.ndarray): soluções unicas obtidas no mapeamento
                    loglikelihood (np.ndarray): valor da função de verossimilhança
    """
    temporal_clusters = np.max(clusters) + 1
    loglikelihood = 0.0
    clusters_temporais = None
    solucoes_unicas = None
    final_clusters = -np.ones(len(clusters))
    final_clusters_offset = 0
    for cluster_id in range(temporal_clusters):

        # extraindo os pontos das soluções
        cluster_solucoes_indexes = np.argwhere(clusters == cluster_id)
        cluster_solucoes = solucoes[cluster_solucoes_indexes]

        # clusterização de distâncias
        (labels,
         centroides,
         distancias_medias,
         detectores) = clusterizacao_DBSCAN3D(cluster_solucoes, eps, min_pts, sistema_cartesiano)

        if len(labels >= 0):

            # construindo um vetor de soluções únicas
            if solucoes_unicas is None:
                solucoes_unicas = centroides
            else:
                final_clusters_offset = len(solucoes_unicas)
                solucoes_unicas = concat_manual(solucoes_unicas, centroides)

            # verossimilhança temporal
            cluster_tempos_medios = tempos[cluster_solucoes_indexes[labels >= 0]] \
                - tempos_medios[cluster_id]
            
            # somando as verossimilhanças

            loglikelihood += (funcao_log_verossimilhanca(cluster_tempos_medios, sigma_t) \
                              + funcao_log_verossimilhanca(distancias_medias, sigma_d))
    
    # substituindo os clusters pelos remapeamento de clusters feito pela clusterização
    # espacial - os clusters finais são as soluções unicas que devem ser substituidas
    # devido à redução do espaço de busca

    clusters = final_clusters

    # enfim, reduzindo o espaço das soluções
    (solucoes,
     solucoes_unicas) = remapeia_solucoes(solucoes,
                                 final_clusters,
                                 solucoes_unicas)

    return solucoes_unicas, loglikelihood
