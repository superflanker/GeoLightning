"""
EELT 7019 - Applied Artificial Intelligence
Meta-heuristics Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
import os
from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_sensor_matrix,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)
from GeoLightning.Simulator.Metrics import (rmse,
                                            mae,
                                            average_mean_squared_error,
                                            mean_location_error,
                                            calcula_prmse,
                                            erro_relativo_funcao_ajuste)
from GeoLightning.Utils.Constants import *
from GeoLightning.Runners.RunnerESO import runner_ESO


def test_runner_ESO(fake_test=False):

    # diretório onde salvar os dados

    basedir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../data/'))

    # recuperando o grupo de sensores

    sensors = get_sensors()

    sensor_tt = get_sensor_matrix(sensors=sensors,
                                  wave_speed=AVG_LIGHT_SPEED,
                                  sistema_cartesiano=False)

    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensores_latlon=sensors,
                                                              margem_metros=3000)

    delta_time = 0.0

    for i in range(len(sensor_tt)):
        for j in range(i, len(sensor_tt[i])):
            if sensor_tt[i, j] > delta_time:
                delta_time = sensor_tt[i, j]

    paper_data = list()

    deltas_d = list()

    deltas_t = list()

    deltas_d_refinado = list()

    deltas_t_refinado = list()

    centroides_espaciais_ea = list()

    centroides_espaciais_irls = list()

    centroides_temporais_ea = list()

    centroides_temporais_irls = list()

    # dados default

    num_events = 1

    runs = 1

    sigma_t = SIGMA_T

    multiplier = 3

    # define se estou rodando na mão o script
    if fake_test:

        num_events = 100

        runs = 100

    # gerando os eventos
    min_alt = 935
    max_alt = 935
    min_time = 10000
    max_time = min_time + num_events * delta_time * multiplier

    for r in range(runs):

        print(f"ESO - Rodada {r}")

        # protagonista da história - eventos
        event_positions, event_times = generate_events(num_events=num_events,
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

        # tudo pronto, rodando o runner

        (sol_centroides_espaciais,
            sol_centroides_temporais,
            sol_centroides_espaciais_refinados,
            sol_centroides_temporais_refinados,
            sol_detectores,
            sol_best_fitness,
            sol_best_fitness_refinado,
            sol_reference,
            delta_d,
            delta_d_refinado,
            delta_t,
            delta_t_refinado,
            execution_time,
            associacoes_corretas) = runner_ESO(event_positions,
                                               event_times,
                                               spatial_clusters,
                                               sensor_tt,
                                               sensor_indexes,
                                               detections,
                                               detection_times,
                                               sensors,
                                               min_alt,
                                               max_alt,
                                               150,
                                               30,
                                               CLUSTER_MIN_PTS,
                                               sigma_t,
                                               AVG_LIGHT_SPEED * sigma_t,
                                               EPSILON_T,
                                               AVG_LIGHT_SPEED,
                                               False)

        assert len(sol_centroides_temporais) == len(event_positions)

        assert len(sol_centroides_espaciais) == len(event_times)

        if fake_test:

            """
                Cálculo de parâmetros para o artigo
            """

            # dados temporais
            crlb_temporal = sigma_t ** 2 / 7.0
            crlb_temporal_rmse = np.sqrt(crlb_temporal)
            crlb_temporal_medio = crlb_temporal
            rmse_temporal = rmse(delta_t)
            rmse_temporal_refinado = rmse(delta_t_refinado)

            prmse_temporal = calcula_prmse(
                rmse_temporal, crlb_temporal_rmse)
            prmse_temporal_refinado = calcula_prmse(rmse_temporal_refinado,
                                                    crlb_temporal_rmse)

            mae_temporal = mae(delta_t)
            mae_temporal_refinado = mae(delta_t_refinado)

            mle_temporal = np.abs(mean_location_error(delta_t))
            mle_temporal_refinado = np.abs(
                mean_location_error(delta_t_refinado))

            amse_temporal = average_mean_squared_error(delta_t)
            amse_temporal_refinado = average_mean_squared_error(
                delta_t_refinado)

            # dados espaciais
            crlb_espacial = ((AVG_LIGHT_SPEED * sigma_t) ** 2) / 7.0
            crlb_espacial_rmse = np.sqrt(crlb_espacial)
            crlb_espacial_medio = crlb_espacial
            rmse_espacial = rmse(delta_d)
            rmse_espacial_refinado = rmse(delta_d_refinado)
            prmse_espacial = calcula_prmse(
                rmse_espacial, crlb_espacial_rmse)
            prmse_espacial_refinado = calcula_prmse(rmse_espacial,
                                                    rmse_espacial_refinado)
            mae_espacial = mae(delta_d)
            mae_espacial_refinado = mae(delta_d_refinado)

            mle_espacial = np.abs(mean_location_error(delta_d))
            mle_espacial_refinado = np.abs(
                mean_location_error(delta_d_refinado))

            amse_espacial = average_mean_squared_error(delta_d)
            amse_espacial_refinado = average_mean_squared_error(
                delta_d_refinado)

            # porcentagem das associações corretas

            correct_association_index = np.mean(associacoes_corretas) * 100

            # adicionando no paper

            paper_data.append([sigma_t/SIGMA_T,
                               crlb_espacial_rmse,
                               rmse_espacial,
                               rmse_espacial_refinado,
                               prmse_espacial,
                               prmse_espacial_refinado,
                               mle_espacial,
                               mle_espacial_refinado,
                               mae_espacial,
                               mae_espacial_refinado,
                               amse_espacial,
                               amse_espacial_refinado,
                               crlb_temporal_rmse,
                               rmse_temporal,
                               rmse_temporal_refinado,
                               mle_temporal,
                               mle_temporal_refinado,
                               mae_temporal,
                               mae_temporal_refinado,
                               amse_temporal,
                               amse_temporal_refinado,
                               correct_association_index,
                               sol_best_fitness,
                               np.abs(sol_best_fitness_refinado),
                               sol_reference,
                               execution_time])

            # armazenando os deltas encontrados

            deltas_d.append(delta_d)

            deltas_t.append(delta_t)

            deltas_d_refinado.append(delta_d_refinado)

            deltas_t_refinado.append(delta_t_refinado)

            centroides_espaciais_ea.append(sol_centroides_espaciais)

            centroides_espaciais_irls.append(
                sol_centroides_espaciais_refinados)

            centroides_temporais_ea.append(sol_centroides_temporais)

            centroides_temporais_irls.append(
                sol_centroides_temporais_refinados)

            print(
                f"Erro espacial => Solução EA: {mle_espacial}, Solução Refinada: {mle_espacial_refinado}")

            print(
                f"Erro temporal => Solução EA: {mle_temporal}, Solução Refinada: {mle_temporal_refinado}")

            print("ESO", [crlb_espacial_rmse,
                          rmse_espacial,
                          rmse_espacial_refinado,
                          prmse_espacial,
                          prmse_espacial_refinado,
                          mle_espacial,
                          mle_espacial_refinado,
                          mae_espacial,
                          mae_espacial_refinado,
                          amse_espacial,
                          amse_espacial_refinado,
                          crlb_temporal_rmse,
                          rmse_temporal,
                          rmse_temporal_refinado,
                          mle_temporal,
                          mle_temporal_refinado,
                          mae_temporal,
                          mae_temporal_refinado,
                          amse_temporal,
                          amse_temporal_refinado,
                          correct_association_index,
                          sol_best_fitness,
                          np.abs(sol_best_fitness_refinado),
                          sol_reference,
                          execution_time])

    # salvando o arquivo
    if fake_test:
        paper_data = np.array(paper_data)
        new_paper_data = np.zeros(paper_data.shape[1])
        for i in range(len(paper_data)):
            new_paper_data += paper_data[i]

        new_paper_data /= len(paper_data)
        print("Final: ", new_paper_data)
        """
        Salvando o arquivo para o paper
        """
        paper_data = np.array(paper_data)
        output_file = os.path.join(basedir, "ESO_results.npy")
        np.save(output_file, new_paper_data)
        print(f"\n>> Resultados salvos em: {output_file}")

        deltas_d = np.array(deltas_d)
        deltas_t = np.array(deltas_t)
        deltas_d_refinado = np.array(deltas_d_refinado)
        deltas_t_refinado = np.array(deltas_t_refinado)
        centroides_espaciais_ea = np.array(centroides_espaciais_ea)
        centroides_espaciais_irls = np.array(centroides_espaciais_irls)
        centroides_temporais_ea = np.array(centroides_temporais_ea)
        centroides_temporais_irls = np.array(centroides_temporais_irls)

        output_file = os.path.join(basedir, "ESO_deltas_distancia.npy")
        np.save(output_file, deltas_d)
        print(f"\n>> Diferenças de Distância salvas em: {output_file}")

        output_file = os.path.join(basedir, "ESO_deltas_tempos.npy")
        np.save(output_file, deltas_t)
        print(f"\n>> Diferenças de Tempo salvas em: {output_file}")

        output_file = os.path.join(
            basedir, "ESO_deltas_distancia_refinado.npy")
        np.save(output_file, deltas_d)
        print(
            f"\n>> Diferenças de Distância Refinadas salvas em: {output_file}")

        output_file = os.path.join(basedir, "ESO_deltas_tempos_refinado.npy")
        np.save(output_file, deltas_t)
        print(f"\n>> Diferenças de Tempo Refinados salvas em: {output_file}")

        output_file = os.path.join(basedir, "ESO_centroides_espaciais_ea.npy")
        np.save(output_file, centroides_espaciais_ea)
        print(f"\n>> Centroides espaciais EA salvos em: {output_file}")

        output_file = os.path.join(
            basedir, "ESO_centroides_espaciais_irls.npy")
        np.save(output_file, centroides_espaciais_irls)
        print(f"\n>> Centroides espaciais IRLS salvos em: {output_file}")

        output_file = os.path.join(basedir, "ESO_centroides_temporais_ea.npy")
        np.save(output_file, centroides_temporais_ea)
        print(f"\n>> Centroides temporais EA salvos em: {output_file}")

        output_file = os.path.join(
            basedir, "ESO_centroides_temporais_irls.npy")
        np.save(output_file, centroides_temporais_irls)
        print(f"\n>> Centroides temporais IRLS salvos em: {output_file}")


if __name__ == "__main__":
    test_runner_ESO(True)
