import numpy as np
from time import perf_counter
from sklearn.cluster import AgglomerativeClustering
from numba import jit

# Função de distância euclidiana para ST-DBSCAN
@jit(nopython=True, cache=True)
def distancia(p1, p2, t1, t2, eps_s, eps_t):
    return np.linalg.norm(p1 - p2) <= eps_s and abs(t1 - t2) <= eps_t

# Implementação ST-DBSCAN simples para benchmarking
@jit(nopython=True, cache=True)
def st_dbscan(pontos, tempos, eps_s, eps_t, min_pts):
    N = pontos.shape[0]
    labels = -np.ones(N, dtype=np.int32)
    cluster_id = 0
    visitado = np.zeros(N, dtype=np.bool_)

    for i in range(N):
        if visitado[i]:
            continue
        visitado[i] = True
        vizinhos = []
        for j in range(N):
            if i != j and distancia(pontos[i], pontos[j], tempos[i], tempos[j], eps_s, eps_t):
                vizinhos.append(j)
        if len(vizinhos) + 1 < min_pts:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            k = 0
            while k < len(vizinhos):
                idx = vizinhos[k]
                if not visitado[idx]:
                    visitado[idx] = True
                    novos_vizinhos = []
                    for j in range(N):
                        if idx != j and distancia(pontos[idx], pontos[j], tempos[idx], tempos[j], eps_s, eps_t):
                            novos_vizinhos.append(j)
                    if len(novos_vizinhos) + 1 >= min_pts:
                        for n in novos_vizinhos:
                            if n not in vizinhos:
                                vizinhos.append(n)
                if labels[idx] == -1:
                    labels[idx] = cluster_id
                elif labels[idx] == -1 or labels[idx] == -2:
                    labels[idx] = cluster_id
                k += 1
            cluster_id += 1
    return labels

# Gerando dados sintéticos
np.random.seed(42)
N = 3000
pontos = np.random.uniform(0, 10000, size=(N, 3))  # coordenadas em metros
tempos = np.random.uniform(0, 0.1, size=N)         # tempos entre 0 e 0.1 s

# Parâmetros para ST-DBSCAN
eps_s = 1000.0   # metros
eps_t = 0.005    # segundos
min_pts = 3

# Benchmark ST-DBSCAN

labels_st = st_dbscan(pontos, tempos, eps_s, eps_t, min_pts)
start_st = perf_counter()

labels_st = st_dbscan(pontos, tempos, eps_s, eps_t, min_pts)
end_st = perf_counter()
tempo_st = end_st - start_st

# Para comparação, usamos Agglomerative com métrica composta
from scipy.spatial.distance import pdist, squareform

# Combina espaço e tempo em uma única matriz com pesos
tempo_normalizado = tempos / eps_t
espaco_normalizado = pontos / eps_s
dados_compostos = np.hstack([espaco_normalizado, tempo_normalizado.reshape(-1, 1)])

# Benchmark Agglomerative
start_aggl = perf_counter()
agg = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, linkage='ward')
labels_aggl = agg.fit_predict(dados_compostos)
end_aggl = perf_counter()
tempo_aggl = end_aggl - start_aggl

import pandas as pd

# Resultados
df_resultados = pd.DataFrame({
    'Algoritmo': ['ST-DBSCAN', 'Agglomerative'],
    'Tempo (s)': [tempo_st, tempo_aggl],
    'Clusters Detectados': [len(np.unique(labels_st[labels_st >= 0])), len(np.unique(labels_aggl))]
})

print(df_resultados)