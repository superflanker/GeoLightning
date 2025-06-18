"""
    EELT 7019 - Inteligência Artificial Aplicada
    Métricas de Performance
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from numpy.linalg import inv
from numba import jit
from GeoLightning.Utils.Constants import SIGMA_D, SIGMA_T


@jit(nopython=True, cache=True, fastmath=True)
def rmse(estimadas: np.ndarray,
         reais: np.ndarray) -> np.float64:
    """
        Calcula o erro quadrático médio (RMSE) entre valores estimados e reais.

        Args:
            estimadas (np.ndarray): vetor de posições estimadas (M x D)
            reais (np.ndarray): vetor de posições reais (M x D)

        Returns:
            np.float64: RMSE
    """
    return np.sqrt(np.mean(np.sum((estimadas - reais) ** 2, axis=1)))


@jit(nopython=True, cache=True, fastmath=True)
def mae(estimados: np.ndarray,
        reais: np.ndarray) -> np.float64:
    """
        Calcula o erro absoluto médio (MAE) entre os valores estimados 
        e os valores reais.

        Args:
            estimados (np.ndarray): vetor de valores estimados
            reais (np.ndarray): vetor de valores reais

        Returns:
            np.float64: MAE
    """
    return np.mean(np.abs(estimados - reais))


@jit(nopython=True, cache=True, fastmath=True)
def average_mean_squared_error(estimados: np.ndarray,
                               reais: np.ndarray) -> np.float64:
    """
        Calcula o erro médio quadrático (AMSE) entre os valores estimados 
        e os valores reais.

        Args:
            estimados (np.ndarray): vetor de valores estimados
            reais (np.ndarray): vetor de valores reais

        Returns:
            np.float64: AMSE 
    """
    return np.mean(np.sum((estimados - reais) ** 2, axis=1))


@jit(nopython=True, cache=True, fastmath=True)
def mean_location_error(estimados: np.ndarray,
                        reais: np.ndarray) -> np.float64:
    """
        Calcula o erro de localização médio (MLE) entre os valores estimados 
        e os valores reais.

        Args:
            estimados (np.ndarray): vetor de posições estimadas
            reais (np.ndarray): vetor de posições reais

        Returns:
            np.float64: MLE
    """
    return np.mean(np.linalg.norm(estimados - reais, axis=1))


@jit(nopython=True, cache=True, fastmath=True)
def calcula_prmse(rmse: float,
                  referencia: float) -> float:
    """
        Calcula o RMSE percentual.

        Args:
            rmse (float): valor do RMSE
            referencia (float): valor de fundo de escala

        Returns:
            float: PRMSE
    """
    return 100.0 * rmse / referencia


@jit(nopython=True, cache=True, fastmath=True)
def acuracia_associacao(associacoes_estimadas: np.ndarray,
                        associacoes_reais: np.ndarray) -> float:
    """
        Calcula a acurácia da associação entre detecções e eventos.

        Args:
            associacoes_estimadas (np.ndarray): vetor de índices estimados 
            associacoes_reais (np.ndarray): vetor de índices reais das associações

        Returns:
            float: acurácia da associação
    """
    return np.mean(associacoes_estimadas == associacoes_reais)


@jit(nopython=True, cache=True, fastmath=True)
def erro_relativo_funcao_ajuste(F_estimado: float,
                                F_referencia: float) -> float:
    """
        Calcula o erro relativo percentual entre a função de ajuste estimada e 
        uma referência (por exemplo, o valor ótimo ou benchmark).

        Args:
            F_estimado (float): valor da função de ajuste
            F_referencia (float): valor de referência

        Returns:
            float: erro relativo percentual
    """
    return np.abs(F_estimado - F_referencia) / np.abs(F_referencia) * 100.0


@jit(nopython=True, cache=True, fastmath=True)
def tempo_execucao(tempo_inicial: float,
                   tempo_final: float) -> float:
    """
        Calcula o tempo total de execução.

        Args:
            tempo_inicial (float): tempo de início (em segundos)
            tempo_final (float): tempo de fim (em segundos)

        Returns:
            float: tempo total em segundos
    """
    return tempo_final - tempo_inicial


@jit(nopython=True, cache=True, fastmath=True)
def calcular_crlb_espacial(sigma_d: float = SIGMA_D,
                           N: int = 7) -> np.ndarray:
    """
        Calcula a matriz CRLB para a estimativa da posição de um evento, 
        considerando variância espacial isotrópica.

        Args:
            sigma_d (float): desvio padrão da medida de distância
            N (int): número de sensores

        Returns:
            np.ndarray: matriz CRLB 3x3 (para [x, y, z])
    """
    return (sigma_d ** 2 / N) * np.eye(3)


@jit(nopython=True, cache=True, fastmath=True)
def calcular_crlb_temporal(sigma_t: float = SIGMA_T,
                           N: int = 7) -> np.ndarray:
    """
        Calcula a CRLB para a estimativa do tempo de origem de um evento.

        Args:
            sigma_t (float): desvio padrão da medida de tempo
            N (int): número de sensores

        Returns:
            np.ndarray: matriz CRLB 1x1 (para t_0)
    """
    return (sigma_t ** 2 / N) * np.eye(1)


@jit(nopython=True, cache=True, fastmath=True)
def calcular_crlb_rmse(crlb: np.ndarray) -> float:
    """
        Calcula a média quadrática do traço da matriz CRLB.

        Args:
            crlb (np.ndarray): matriz CRLB

        Returns:
            float: valor médio quadrático (CRLB_RMSE)
    """
    return np.sqrt(np.trace(crlb @ crlb)) / crlb.shape[0]


@jit(nopython=True, cache=True, fastmath=True)
def calcular_mean_crlb(crlb: np.ndarray) -> float:
    """
        Calcula a média das variâncias na matriz CRLB.

        Args:
            crlb (np.ndarray): matriz CRLB

        Returns:
            float: média das variâncias
    """
    return np.trace(crlb) / crlb.shape[0]
