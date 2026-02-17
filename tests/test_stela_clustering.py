"""
EELT 7019 - Applied Artificial Intelligence
Clustering Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_sensor_matrix,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)

from GeoLightning.Utils.Constants import *

from GeoLightning.Stela.Stela import stela_phase_one

import scienceplots

from scipy.signal import savgol_filter

plt.style.use(['science'])

plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.titlesize': 8,
    'legend.loc': 'upper left',   # ou 'best', 'lower right', etc.
    'legend.frameon': False,
    'legend.handlelength': 2.0,
    'legend.borderaxespad': 0.4,
})


def test_clustering_stela(fake_test=False):

    # diretório onde salvar os dados

    basedir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../data/'))

    # recuperando o grupo de sensores
    sensors = get_sensors()
    sensor_tt = get_sensor_matrix(sensors, AVG_LIGHT_SPEED, False)
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    from scipy.spatial import ConvexHull

    # sensores: array [[lat, lon], ...]
    hull = ConvexHull(sensors[:, :2])
    vertices_hull = sensors[hull.vertices, :2] # Apenas os sensores da borda, 

    # gerando os eventos
    min_alt = 935
    max_alt = 935
    min_time = 10000

    delta_time = 0.0

    for i in range(len(sensor_tt)):
        for j in range(i, len(sensor_tt[i])):
            if sensor_tt[i, j] > delta_time:
                delta_time = sensor_tt[i, j]

    # dados default

    num_events = 100

    time_multipliers = [10]

    runs = 1

    x = list()

    y = list()

    # define se estou rodando na mão o script
    if fake_test:

        num_events = 10000

        """time_multipliers = [0.1, 0.2, 0.3, 0.4, 0.5,
                            1, 2, 3, 4, 5, 
                            6, 7, 8, 9, 10, 
                            11, 12, 13,  14, 15,
                            16, 17, 18, 19, 20]"""
        
        time_multipliers = np.linspace(0, 2, 2000)

        runs = 10

    for time_multiplier in time_multipliers:

        max_time = min_time + num_events * time_multiplier * delta_time 

        acc_eff = 0.0

        for _ in range(runs):

            # protagonista da história - eventos
            event_positions, event_times = generate_events(num_events=num_events,
                                                           vertices_hull=vertices_hull,
                                                           min_lat=min_lat,
                                                           max_lat=max_lat,
                                                           min_lon=min_lon,
                                                           max_lon=max_lon,
                                                           min_alt=min_alt,
                                                           max_alt=max_alt,
                                                           min_time=min_time,
                                                           max_time=max_time,
                                                           fixed_seed=False)

            # gerando as detecções
            (detections,
                detection_times,
                n_event_positions,
                n_event_times,
                distances,
                sensor_indexes,
                spatial_clusters) = generate_detections(event_positions=event_positions,
                                                        event_times=event_times,
                                                        sensor_positions=sensors,
                                                        simulate_complete_detections=True,
                                                        fixed_seed=False,
                                                        min_pts=CLUSTER_MIN_PTS)

            # tudo pronto, rodando a clusterização

            (tempos_ordenados,
             indices_sensores_ordenados,
             clusters_espaciais,
             ordered_indexes) = stela_phase_one(tempos_de_chegada=detection_times,
                                                indices_sensores=sensor_indexes,
                                                sensor_tt=sensor_tt,
                                                epsilon_t=EPSILON_T,
                                                min_pts=CLUSTER_MIN_PTS)
            
            len_clusterizados = len(
                np.unique(clusters_espaciais[clusters_espaciais >= 0]))
            len_reais = len(event_positions)
            acc_eff += len_clusterizados/len_reais

        if fake_test:
            acc_eff /= runs
            print(time_multiplier * delta_time, 100 * acc_eff)

            x.append(time_multiplier * delta_time)
            y.append(100 * acc_eff)

    if fake_test:

        # Parâmetros
        window_length = 25  # deve ser ímpar e menor ou igual ao tamanho de y
        polyorder = 2      # grau do polinômio ajustado em cada janela

        # Aplicando o filtro de suavização
        y_savgol = savgol_filter(y,
                                 window_length=window_length,
                                 polyorder=polyorder)

        # Plotando os resultados

        plt.plot(x,
                 y_savgol,
                 '-',
                 color='black',
                 label='Observed Separation')

        plt.plot(x,
                 100 * np.ones(len(x)),
                 linestyle='--',
                 color='gray',
                 linewidth=1.0,
                 label=r'Reference ($100\%$)')

        plt.xlabel(r'Mean Time Between Events ($s/\text{event}$)')
        plt.ylabel(r'Separation Efficiency ($\%$)')
        plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
        plt.tight_layout()
        filename = os.path.join(basedir, "images/clustering_efficiency.png")
        plt.savefig(filename, dpi=600)  # é pro IEEE, certo?


if __name__ == "__main__":
    test_clustering_stela(True)
