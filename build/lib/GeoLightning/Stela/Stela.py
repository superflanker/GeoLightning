"""
    EELT 7019 - Inteligência Artificial Aplicada  
    Algoritmo STELA - Spatio-Temporal Event Likelihood Assignment  
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>

    Este módulo implementa o algoritmo STELA, responsável por realizar a 
    associação espaço-temporal entre detecções multissensoriais e eventos 
    físicos simulados, como descargas atmosféricas ou fontes acústicas impulsivas.  

    O algoritmo combina estratégias de clusterização espacial e temporal, 
    além de calcular uma função de verossimilhança baseada na consistência 
    entre o tempo de chegada dos sinais e a posição estimada dos eventos.  

    O STELA é projetado para refinar soluções iniciais de multilateração, 
    ajustar os limites de busca para meta-heurísticas e identificar agrupamentos 
    plausíveis de detecções que correspondam a eventos reais.

    Este pipeline é compatível com metodologias de localização baseadas 
    em TOA (Time-of-Arrival), sendo aplicável a contextos como sensoriamento geofísico, 
    radiofrequência, acústica submarina e astronomia transiente.

"""


import numpy as np
from numba import jit
from GeoLightning.Utils.Constants import SIGMA_D, \
    EPSILON_D, \
    EPSILON_T, \
    LIMIT_D, \
    CLUSTER_MIN_PTS, \
    MAX_DISTANCE
from GeoLightning.Utils.Utils import computa_tempos_de_origem
from GeoLightning.Stela.Deprecated.TemporalClustering import clusterizacao_temporal_stela
from GeoLightning.Stela.Deprecated.SpatialClustering import clusterizacao_espacial_stela
from GeoLightning.Stela.Bounds import gera_limites


@jit(nopython=True, cache=True, fastmath=True)
def stela(solucoes: np.ndarray,
          tempos_de_chegada: np.ndarray,
          pontos_de_deteccao: np.ndarray,
          clusters_espaciais: np.ndarray,
          sistema_cartesiano: bool = False,
          sigma_d: np.float64 = SIGMA_D,
          epsilon_t: np.float64 = EPSILON_T,
          epsilon_d: np.float64 = EPSILON_D,
          limit_d: np.float64 = LIMIT_D,
          max_d: np.float64 = MAX_DISTANCE,
          min_pts: np.int32 = CLUSTER_MIN_PTS) -> tuple:
    """
        Algoritmo STELA - Spatio-Temporal Event Likelihood Assignment.

        Esta função executa a fase de associação e filtragem de eventos com base 
        na consistência espaço-temporal entre as detecções multissensoriais 
        e as soluções candidatas geradas por multilateração.

        O algoritmo aplica clusterização temporal para inferir tempos de origem 
        estimados, seguida de uma nova clusterização espacial com base na 
        compatibilidade das detecções, otimizando uma função de log-verossimilhança.

        Parâmetros:
            solucoes (np.ndarray): Array de forma (N, 3), contendo as posições 
                candidatas dos eventos em coordenadas geográficas ou cartesianas.
            tempos_de_chegada (np.ndarray): Array de forma (M,), contendo os 
                tempos absolutos de chegada dos sinais em cada sensor.
            pontos_de_deteccao (np.ndarray): Array de forma (M, 3), com as posições 
                dos sensores (em mesmas coordenadas que `solucoes`).
            clusters_espaciais (np.ndarray): Array (N,), contendo o identificador 
                de cluster espacial de cada solução candidata.
            sistema_cartesiano (bool): Indica se os dados estão em coordenadas 
                cartesianas (True) ou geográficas (False). Padrão: False.
            sigma_d (float): Desvio padrão espacial (usado na verossimilhança).
            epsilon_t (float): Tolerância temporal para a clusterização temporal.
            epsilon_d (float): Tolerância espacial para a clusterização espacial.
            limit_d (float): Raio máximo para o cálculo dos limites de busca 
                (bounding box) das meta-heurísticas.
            max_d (float): Distância máxima permitida entre eventos e sensores.
            min_pts (int): Número mínimo de pontos para formar um cluster (DBSCAN).

        Retorna:
            tuple:
                lb (np.ndarray): Vetor com os limites inferiores para otimização.
                ub (np.ndarray): Vetor com os limites superiores para otimização.
                centroides (np.ndarray): Coordenadas médias dos eventos detectados.
                detectores (np.ndarray): Índices dos sensores associados.
                clusters_espaciais (np.ndarray): Rótulos atualizados dos clusters.
                novas_solucoes (np.ndarray): Soluções refinadas espacialmente.
                verossimilhanca (float): Valor agregado da verossimilhança.

        Observações:
            - A função é otimizada com Numba para alto desempenho.
            - O modelo é compatível com cenários de múltiplos eventos e sensores.
            - Pode ser utilizado como fase de pré-processamento para otimizações 
            com algoritmos genéticos, swarm intelligence e outros métodos globais.
    """

    # primeiro passo - clusters temporais
    tempos_de_origem = computa_tempos_de_origem(solucoes,
                                                clusters_espaciais,
                                                tempos_de_chegada,
                                                pontos_de_deteccao,
                                                sistema_cartesiano)

    clusters_temporais = clusterizacao_temporal_stela(tempos_de_origem,
                                                      epsilon_t,
                                                      min_pts)

    verossimilhanca = 0.0
    # segundo passo - clusterização espacial e cálculo da função de fitness
    # adicionado: calculamos o remapeamento espacial aqui também

    (centroides,
     detectores,
     solucoes_unicas,
     clusters_espaciais,
     novas_solucoes,
     loglikelihood) = clusterizacao_espacial_stela(solucoes,
                                                   clusters_temporais,
                                                   tempos_de_origem,
                                                   epsilon_d,
                                                   sigma_d,
                                                   min_pts,
                                                   sistema_cartesiano)

    verossimilhanca += loglikelihood

    # calculando os limites para o algoritmo meta-heurístico

    lb, ub = gera_limites(novas_solucoes,
                          solucoes_unicas,
                          limit_d,
                          max_d,
                          sistema_cartesiano)

    # tudo pronto, retornando
    return (lb,
            ub,
            centroides,
            detectores,
            clusters_espaciais,
            novas_solucoes,
            verossimilhanca)


if __name__ == "__main__":

    num_events = [2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000,
                  2000, 3000, 4000, 5000, 6000,
                  7000, 8000, 9000, 10000]

    from time import perf_counter

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

        event_positions = np.load(file_event_positions)
        event_times = np.load(file_event_times)
        pontos_de_deteccao = np.load(file_detections)
        tempos_de_chegada = np.load(file_detections_times)
        solucoes = np.load(file_n_event_positions)
        # spatial_clusters = np.load(file_spatial_clusters)
        spatial_clusters = np.cumsum(
            np.ones(len(solucoes), dtype=np.int32)) - 1
        start_st = perf_counter()

        (lb,
         ub,
         centroides,
         detectores,
         clusters_espaciais,
         novas_solucoes,
         verossimilhanca) = stela(solucoes,
                                  tempos_de_chegada,
                                  pontos_de_deteccao,
                                  spatial_clusters,
                                  sistema_cartesiano=False)
        end_st = perf_counter()

        print(f"Elapsed time: {end_st - start_st:.6f} seconds")

        print(verossimilhanca)
        print(clusters_espaciais)
        print(lb)
        print(ub)

        # a repetição deve ser com um conjunto bem menor de soluções

        start_st = perf_counter()

        (lb,
         ub,
         centroides,
         detectores,
         clusters_espaciais,
         novas_solucoes,
         verossimilhanca) = stela(solucoes,
                                  tempos_de_chegada,
                                  pontos_de_deteccao,
                                  spatial_clusters,
                                  sistema_cartesiano=False)

        end_st = perf_counter()

        print(f"Elapsed time: {end_st - start_st:.6f} seconds")

        print(verossimilhanca)
        print(clusters_espaciais)
        print(lb)
        print(ub)
