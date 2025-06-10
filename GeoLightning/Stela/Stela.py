"""
    EELT 7019 - Inteligência Artificial Aplicada
    Algoritmo STELA - Algoritmo Principal
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

import numpy as np
from numba import jit
from GeoLightning.Utils.Constants import SIGMA_D, \
    SIGMA_T, \
    EPSILON_D, \
    EPSILON_T, \
    CLUSTER_MIN_PTS
from GeoLightning.Utils.Utils import computa_tempos_de_origem
from .TemporalClustering import clusterizacao_temporal_stela
from .SpatialClustering import clusterizacao_espacial_stela
from .LogLikelihood import funcao_log_verossimilhanca
from .Entropy import calcular_entropia_local


def stela(solucoes: np.ndarray,
          tempos_de_chegada: np.ndarray,
          pontos_de_deteccao: np.ndarray,
          clusters_espaciais: np.ndarray,
          sistema_cartesiano: bool = False,
          p_lambda: np.float64 = 1.0,
          sigma_t: np.float64 = SIGMA_T,
          sigma_d: np.float64 = SIGMA_D,
          epsilon_t: np.float64 = EPSILON_T,
          epsilon_d: np.float64 = EPSILON_D,
          min_pts: np.int32 = CLUSTER_MIN_PTS) -> np.float64:
    """
        Algoritmo STELA - Spatio-Temporal Event Likelihood Assignment

        Avalia simultaneamente a verossimilhança espacial e 
        temporal de eventos detectados por múltiplas estações, 
        agrupando e refinando as soluções candidatas de multilateração 
        com base em consistência espaço-temporal.

        Este algoritmo aplica um critério de compatibilidade entre as 
        soluções geradas por diferentes agrupamentos, associando detecções 
        a possíveis eventos físicos, considerando tanto a distância 
        geodésica quanto os atrasos esperados de chegada.

        Args:
            solucoes (np.ndarray): Array de (N, 3), contendo 
                as coordenadas geográficas (latitude, longitude, altitude) 
                das soluções candidatas a eventos.
            tempos_de_chegada (np.ndarray): Array de forma (M,), contendo 
                os tempos absolutos (em segundos ou outra unidade compatível) 
                das chegadas em cada estação ou detector.
            pontos_de_deteccao (np.ndarray): Array de forma (M, 3), contendo 
                as coordenadas (latitude, longitude, altitude) das estações 
                que detectaram os sinais.
            clusters_espaciais (np.ndarray): Array de forma (N,), associando 
                cada solução a um identificador de cluster espacial.
            p_lambda (np.float64): fator de penalização arbitrário
            sigma_t (np.float64): desvio padrão temporal
            sigma_d (np.float64): desvio padrão espacial
            epsilon_t (np.float64): tolerância máxima em segundos
            eps (np.float64): tolerância máxima (janela espacial) 
                            para definição de vizinhança
            min_pts (np.int32): número mínimo de pontos para formar um cluster

        Returns:
            verossimilhança (np.float64): avaliação da função de verosimilhança
        Observações:
            - Este algoritmo é adequado para aplicações em geolocalização 
            de descargas atmosféricas, eventos sísmicos ou qualquer fenômeno 
            pontual detectável por sensores distribuídos.
    """

    # primeiro passo - clusters temporais
    tempos_de_origem = computa_tempos_de_origem(solucoes,
                                                clusters_espaciais,
                                                tempos_de_chegada,
                                                pontos_de_deteccao,
                                                sistema_cartesiano)

    (clusters_temporais,
     tempos_medios,
     _) = clusterizacao_temporal_stela(tempos_de_origem,
                                                epsilon_t,
                                                min_pts)
    
    # verossimilhança dos tempos não clusterizados - entropia
    temps_nao_clusterizados = tempos_de_origem[clusters_temporais == 1]
    verossimilhanca = 0.0
    if len(temps_nao_clusterizados) > 0:
        verossimilhanca -= p_lambda * calcular_entropia_local(temps_nao_clusterizados)

    # segundo passo - clusterização espacial e cálculo da função de fitness
    # adicionado: calculamos o remapeamento espacial aqui também
    (clusters_espaciais,
     log_likelihood) = clusterizacao_espacial_stela(solucoes,
                                                    clusters_temporais,
                                                    tempos_de_origem,
                                                    tempos_medios,
                                                    epsilon_d,
                                                    sigma_t,
                                                    sigma_d,
                                                    min_pts,
                                                    sistema_cartesiano)
    verossimilhanca += log_likelihood
    
    # tudo pronto, retornando
    return verossimilhanca
