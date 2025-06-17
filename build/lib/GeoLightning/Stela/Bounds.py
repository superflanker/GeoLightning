"""
    EELT 7019 - Inteligência Artificial Aplicada
    Limiares dinâmicos - para os pontos da solução
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
from numba import jit
import numpy as np
from GeoLightning.Utils.Constants import EPSILON_D, \
    MAX_DISTANCE, \
    R_LAT


@jit(nopython=True, cache=True, fastmath=True)
def gera_limites(pontos_clusterizados: np.ndarray,
                 clusters: np.ndarray,
                 raio_metros: np.float64 = EPSILON_D,
                 raio_maximo: np.float64 = MAX_DISTANCE,
                 sistema_cartesiano: bool = False) -> tuple:
    """
    Versão otimizada com Numba da geração de limites de busca local ao redor de pontos clusterizados.

    Args:
        pontos_clusterizados (np.ndarray): matriz (N,3) com colunas [lat, lon, alt] em graus e metros
        clusters (np.ndarray): os clusters da solução reduzida
        raio_metros (np.float64): raio de busca em metros ao redor de cada ponto de solução única
        raio_maximo (np.float64): raio máximo em metros ao redor de cada ponto de solução não única
        sistema_cartesiano (bool): indicador de sistema cartesiano

    Returns:
        tuple =>
                lb, ub (np.ndarray): limites aplicáves à solução
    """
    n = pontos_clusterizados.shape[0]

    ub = np.zeros((n, 3), dtype=np.float64)

    lb = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        lat = pontos_clusterizados[i, 0]
        lon = pontos_clusterizados[i, 1]
        alt = pontos_clusterizados[i, 2]

        d_raio = raio_metros
        if clusters[i] == -1:
            d_raio = raio_maximo

        if sistema_cartesiano:
            dlat = d_raio
            dlon = d_raio
            dalt = d_raio
        else:
            dlat = d_raio / R_LAT
            dlon = d_raio / (R_LAT * np.cos(np.radians(lat)))
            dalt = 5 * d_raio
            if dalt > 30000:
                dalt = 30000

        lb[i, 0] = lat - dlat
        lb[i, 1] = lon - dlon
        lb[i, 2] = alt - dalt

        if not sistema_cartesiano:

            if lb[i, 2] < 0:
                lb[i, 2] = 0

        ub[i, 0] = lat + dlat
        ub[i, 1] = lon + dlon
        ub[i, 2] = alt + dalt

    return lb.flatten(), ub.flatten()


if __name__ == "__main__":

    # Recriar os pontos exemplo
    pontos_exemplo = np.array([
        [-25.0, -49.0, 800.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.1, -49.1, 1000.0],
        [-25.2, -49.2, 1200.0]
    ])

    solucoes_unicas = np.array([1, 1, 1, 0, 0, 0, 0, 0])

    # Teste de verificação
    lb, ub = gera_limites_esfericos(pontos_exemplo, solucoes_unicas)
    print(lb, ub)
