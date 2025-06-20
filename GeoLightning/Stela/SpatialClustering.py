"""
Spatial Clustering
==================

Spatio-temporal Clustering Based on Temporal Hierarchies
---------------------------------------------------------

Summary
-------
This module performs the spatial clustering phase of the STELA algorithm,
based on temporally grouped candidate events. Using a hierarchical DBSCAN-like
approach, this stage aggregates spatio-temporal clusters, computes centroids, 
and evaluates the spatial likelihood.

Author
------
Augusto Mathias Adams <augusto.adams@ufpr.br>

Notes
-----
This module is part of the activities of the discipline  
EELT 7019 - Applied Artificial Intelligence, Federal University of Paraná (UFPR), Brazil.

Dependencies
------------
- numpy
- numba
"""
from numba import jit
import numpy as np
from GeoLightning.Utils.Constants import SIGMA_T, \
    SIGMA_D, \
    EPSILON_D, \
    CLUSTER_MIN_PTS
from GeoLightning.Stela.DBSCAN3D import clusterizacao_DBSCAN3D
from GeoLightning.Stela.LogLikelihood import funcao_log_verossimilhanca
from GeoLightning.Stela.Dimensions import remapeia_solucoes, \
    remapeia_solucoes_unicas
from GeoLightning.Stela.Common import calcula_distancias_ao_centroide_ak, \
    calcular_centroides_ak
from GeoLightning.Stela.Entropy import calcular_entropia_local


@jit(nopython=True, cache=True, fastmath=True)
def clusterizacao_espacial_stela(solucoes: np.ndarray,
                                 clusters: np.ndarray,
                                 tempos_de_origem: np.ndarray,
                                 eps: np.float64 = EPSILON_D,
                                 sigma_d: np.float64 = SIGMA_D,
                                 min_pts: np.int32 = CLUSTER_MIN_PTS,
                                 sistema_cartesiano: bool = False) -> tuple:
    """
    Spatial clustering stage of the STELA algorithm.

    This function groups spatially-close events based on the labels 
    from a prior temporal clustering step. It applies DBSCAN in 3D 
    for each temporal cluster and computes associated centroids, 
    updated solutions, and the log-likelihood score.

    Parameters
    ----------
    solucoes : np.ndarray
        Array of shape (N, 3) containing candidate event positions.
    clusters : np.ndarray
        Vector of temporal cluster labels assigned to each candidate.
    tempos_de_origem : np.ndarray
        Origin times for each candidate event.
    eps : float, optional
        Maximum distance threshold for spatial neighborhood in clustering (default=EPSILON_D).
    sigma_d : float, optional
        Spatial standard deviation for likelihood computation (default=SIGMA_D).
    min_pts : int, optional
        Minimum number of points to form a valid cluster (default=CLUSTER_MIN_PTS).
    sistema_cartesiano : bool, optional
        If True, uses Cartesian distance; otherwise, geodetic distance is applied (default=False).

    Returns
    -------
    tuple
        centroides : np.ndarray
            Coordinates of spatial centroids for each final cluster.
        detectores : np.ndarray
            Detector mask associated with each cluster.
        solucoes_unicas : np.ndarray
            Mapped labels of unique spatial clusters.
        final_clusters : np.ndarray
            Final cluster labels after spatial refinement.
        novas_solucoes : np.ndarray
            Updated solution array based on centroid association.
        loglikelihood : float
            Log-likelihood value of the current configuration.
    """
    temporal_clusters = np.max(clusters) + 1
    loglikelihood = 0.0
    final_clusters_base_index = 0  # indexador base dos clusters finais
    final_clusters = -np.ones(len(clusters), dtype=np.int32)

    for cluster_id in range(temporal_clusters):

        # extraindo as soluçũes indicadas pelo cluster

        cluster_solucoes_indexes = np.argwhere(
            clusters == cluster_id).flatten()

        # extraindo as soluçũes indicadas pelo cluster

        cluster_solucoes = np.zeros((cluster_solucoes_indexes.shape[0],
                                     solucoes.shape[1]),
                                    dtype=solucoes.dtype)

        for i in range(len(cluster_solucoes_indexes)):
            cluster_solucoes[i] = solucoes[cluster_solucoes_indexes[i]]

        # clusterização espacial
        labels = clusterizacao_DBSCAN3D(cluster_solucoes,
                                        eps,
                                        min_pts,
                                        sistema_cartesiano)

        if len(labels[labels >= 0]) > 0:

            # agora, vamos refazer a lista de clusters em final_clusters
            for i in range(len(labels)):
                if labels[i] >= 0:
                    final_clusters[cluster_solucoes_indexes[i]
                                   ] = final_clusters_base_index + labels[i]
            # base update
            final_clusters_base_index += np.max(labels) + 1

    # agora sim, posso calcular tempos e distâncias médias, que tal?

    centroides, detectores = calcular_centroides_ak(solucoes,
                                                    final_clusters)


    novas_solucoes = remapeia_solucoes(
        solucoes, final_clusters, centroides
    )

    solucoes_unicas = remapeia_solucoes_unicas(final_clusters)

    distancias = calcula_distancias_ao_centroide_ak(novas_solucoes,
                                                    final_clusters,
                                                    centroides,
                                                    sistema_cartesiano)

    loglikelihood = calcular_entropia_local(tempos_de_origem[final_clusters == -1]) \
        + funcao_log_verossimilhanca(distancias, sigma_d)

    return (centroides,
            detectores,
            solucoes_unicas,
            final_clusters,
            novas_solucoes,
            loglikelihood)


if __name__ == "__main__":

    from GeoLightning.Stela.TemporalClustering import clusterizacao_temporal_stela
    from GeoLightning.Utils.Utils import computa_tempos_de_origem
    from time import perf_counter

    num_events = [2, 5, 10, 15, 20, 25,
                  30, 100, 500, 800, 1000]

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
        start_st = perf_counter()

        tempos_de_origem = computa_tempos_de_origem(n_event_positions,
                                                    spatial_clustering,
                                                    detection_times,
                                                    detection_positions)

        labels = clusterizacao_temporal_stela(
            tempos_de_origem)

        (centroides,
         detectores,
         solucoes_unicas,
         final_clusters,
         novas_solucoes,
         loglikelihood) = clusterizacao_espacial_stela(n_event_positions,
                                                       labels.astype(dtype=np.int32),
                                                       tempos_de_origem)

        end_st = perf_counter()

        print(f"Elapsed time: {end_st - start_st:.6f} seconds")

        print(len(np.unique(final_clusters)), loglikelihood)
        print(len(n_event_positions), len(novas_solucoes))
