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

    # gerando os eventos
    min_alt = 935
    max_alt = 935
    min_time = 10000

    paper_data = list()

    deltas_d = list()

    deltas_t = list()

    # dados default

    num_events = 1

    max_times = [50]

    runs = 1

    x = list()

    y = list()

    # define se estou rodando na mão o script
    if fake_test:

        num_events = 1000

        max_times = np.arange(10, 212, 2)

        runs = 100

    for max_time in max_times:

        acc_eff = 0.0

        for _ in range(runs):

            # o protagonista da história - os eventos

            event_positions, event_times = generate_events(num_events,
                                                           min_lat,
                                                           max_lat,
                                                           min_lon,
                                                           max_lon,
                                                           min_alt,
                                                           max_alt,
                                                           min_time,
                                                           min_time + max_time)

            # gerando as detecções
            (detections,
             detection_times,
             n_event_positions,
             n_event_times,
             distances,
             sensor_indexes,
             spatial_clusters) = generate_detections(event_positions,
                                                     event_times,
                                                     sensors)

            # tudo pronto, rodando a clusterização

            clusters_espaciais = stela_phase_one(detection_times,
                                                 sensor_indexes,
                                                 sensor_tt,
                                                 EPSILON_T/1.5,
                                                 CLUSTER_MIN_PTS)

            len_clusterizados = len(
                np.unique(clusters_espaciais[clusters_espaciais >= 0]))
            len_reais = len(event_positions)
            acc_eff += len_clusterizados/len_reais

        if fake_test:
            acc_eff /= runs
            print(max_time, 100 * acc_eff)

            x.append(max_time/num_events)
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

        """plt.plot(x, 
                 y, 
                 color='black', 
                 label='Observed Separation')"""
        
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
                 label='Reference ($100\%$)')

        plt.xlabel('Mean Time Between Events ($s/\\text{event}$)')
        plt.ylabel('Separation Efficiency (\%)')
        plt.legend(loc='best', ncol=1, shadow=True, fancybox=True)
        plt.tight_layout()
        filename = os.path.join(basedir, "clustering_efficiency.png")
        plt.savefig(filename, dpi=600) # é pro IEEE, certo?


if __name__ == "__main__":
    test_clustering_stela(True)
