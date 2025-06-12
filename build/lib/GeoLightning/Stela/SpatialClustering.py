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
from GeoLightning.Stela.DBSCAN3D import clusterizacao_DBSCAN3D
from GeoLightning.Stela.LogLikelihood import funcao_log_verossimilhanca
from GeoLightning.Stela.Dimensions import remapeia_solucoes


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
                    final_clusters (np.ndarray): a clusterização espaço-temporal final
                    loglikelihood (np.ndarray): valor da função de verossimilhança
    """
    temporal_clusters = np.max(clusters) + 1
    loglikelihood = 0.0
    max_centroides = solucoes.shape[0]
    solucoes_unicas = np.empty((max_centroides, 3), dtype=np.float64)
    su_index = 0  # índice para inserção incremental
    final_clusters_base_index = 0 # indexador base dos clusters finais
    final_clusters = -np.ones(len(clusters), dtype=np.float64)

    for cluster_id in range(temporal_clusters):

        cluster_solucoes_indexes = np.argwhere(
            clusters == cluster_id).flatten()

        # extraindo as soluçũes indicadas pelo cluster

        cluster_solucoes = np.zeros((cluster_solucoes_indexes.shape[0],
                                     solucoes.shape[1]), dtype=solucoes.dtype)
        
        clusters_solucoes_tempos = np.zeros(cluster_solucoes_indexes.shape[0], 
                                            dtype=tempos.dtype)
        
        for i in range(len(cluster_solucoes_indexes)):
            cluster_solucoes[i] = solucoes[cluster_solucoes_indexes[i]]
            clusters_solucoes_tempos[i] = tempos[cluster_solucoes_indexes[i]]

        # clusterização espacial
        (labels,
         centroides,
         distancias_medias,
         _) = clusterizacao_DBSCAN3D(
            cluster_solucoes, eps, min_pts, sistema_cartesiano
        )

        if len(labels) > 0:

            # guardando os centróides
            for i in range(centroides.shape[0]):
                solucoes_unicas[su_index] = centroides[i]
                su_index += 1

            # retirando as posições onde temos labels >= 0
            clusters_espaciais_validos = np.argwhere(
                        labels >= 0).flatten()

            # extraindo os tempos para cálculo da verossimilhança

            cluster_tempos = np.zeros(clusters_espaciais_validos.shape[0],
                                      dtype=clusters_solucoes_tempos.dtype)

            for i in range(len(clusters_espaciais_validos)):
                cluster_tempos[i] = clusters_solucoes_tempos[clusters_espaciais_validos[i]]

            # função de verossimilhança

            cluster_tempos_medios = cluster_tempos - tempos_medios[cluster_id]

            loglikelihood += (
                funcao_log_verossimilhanca(cluster_tempos_medios, sigma_t)
                + funcao_log_verossimilhanca(distancias_medias, sigma_d)
            )

            # agora, vamos refazer a lista de clusters em final_clusters
            for i in range(len(labels)):
                if labels[i] >= 0:
                    final_clusters[cluster_solucoes_indexes[i]
                                   ] = final_clusters_base_index + labels[i]
            # base update
            final_clusters_base_index += su_index

    # recorta a matriz final com apenas os centroides adicionados
    solucoes_unicas = solucoes_unicas[:su_index]

    solucoes = remapeia_solucoes(
        solucoes, final_clusters, solucoes_unicas
    )

    return solucoes_unicas, final_clusters, solucoes, loglikelihood

if __name__ == "__main__":
    
    from GeoLightning.Stela.TemporalClustering import clusterizacao_temporal_stela
    from GeoLightning.Utils.Utils import computa_tempos_de_origem
    num_events = [2, 5, 10, 15, 20, 25, 30, 100, 500, 800, 1000, 2000, 3000]

    for i in range(len(num_events)):

        print("Events: {:d}".format(num_events[i]))
        
        file_detections = "../../data/static_constellation_detections_{:06d}.npy".format(
            num_events[i])

        file_detections_times = "../../data/static_constelation_detection_times_{:06d}.npy".format(
            num_events[i])

        file_event_positions = "../../data/static_constelation_event_positions_{:06d}.npy".format(
            num_events[i])

        file_event_times = "../../data/static_constelation_event_times_{:06d}.npy".format(
            num_events[i])

        file_n_event_positions = "../../data/static_constelation_n_event_positions_{:06d}.npy".format(
            num_events[i])

        file_n_event_times = "../../data/static_constelation_n_event_times_{:06d}.npy".format(
            num_events[i])

        file_distances = "../../data/static_constelation_distances_{:06d}.npy".format(
            num_events[i])
        
        file_spatial_clusters = "../../data/static_constelation_spatial_clusters_{:06d}.npy".format(
            num_events[i])

        n_event_times = np.load(file_n_event_times)
        n_event_positions = np.load(file_n_event_positions)
        detection_times = np.load(file_detections_times)
        detection_positions = np.load(file_detections)
        spatial_clustering = np.load(file_spatial_clusters)

        # calculando os tempos de origem

        tempos_de_origem = computa_tempos_de_origem(n_event_positions,
                                                    spatial_clustering,
                                                    detection_times,
                                                    detection_positions)
        
        labels, tempos_medios, detectores = clusterizacao_temporal_stela(tempos_de_origem)

        (solucoes_unicas,
         final_clusters, 
         solucoes, 
         loglikelihood) = clusterizacao_espacial_stela(n_event_positions,
                                                       labels,
                                                       n_event_times,
                                                       tempos_medios)

        print(len(np.unique(final_clusters)))
