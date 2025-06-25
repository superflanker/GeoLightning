"""
EELT 7019 - Applied Artificial Intelligence
Meta-heuristics Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
import os
from GeoLightning.Simulator.Simulator import (get_sensors,
                                              get_random_sensors,
                                              get_lightning_limits,
                                              generate_detections,
                                              generate_events)
from GeoLightning.Simulator.Metrics import (rmse,
                                            mae,
                                            average_mean_squared_error,
                                            mean_location_error,
                                            calcula_prmse,
                                            erro_relativo_funcao_ajuste,
                                            calcular_crlb_espacial,
                                            calcular_crlb_temporal,
                                            calcular_crlb_rmse,
                                            calcular_mean_crlb)
from GeoLightning.Utils.Constants import *
from GeoLightning.Runners.RunnerESO import runner_ESO


def test_runner_ESO(fake_test=False):

    # diretório onde salvar os dados

    basedir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../data/'))

    # recuperando o grupo de sensores
    sensors = get_sensors()
    min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

    # gerando os eventos
    min_alt = 935
    max_alt = 935
    min_time = 10000
    max_time = min_time + 72 * 3600

    paper_data = list()

    # dados default

    num_events = [1]

    desired_sigma_t = [SIGMA_T]

    # define se estou rodando na mão o script
    if fake_test:
        num_events = [10000]

        desired_sigma_t = [SIGMA_T,
                           10 * SIGMA_T,
                           20 * SIGMA_T,
                           30 * SIGMA_T,
                           40 * SIGMA_T,
                           50 * SIGMA_T,
                           60 * SIGMA_T,
                           70 * SIGMA_T,
                           80 * SIGMA_T,
                           90 * SIGMA_T,
                           100 * SIGMA_T,
                           200 * SIGMA_T,
                           300 * SIGMA_T,
                           400 * SIGMA_T,
                           500 * SIGMA_T,
                           600 * SIGMA_T,
                           700 * SIGMA_T,
                           800 * SIGMA_T,
                           900 * SIGMA_T,
                           1000 * SIGMA_T]

    for i in range(len(desired_sigma_t)):

        sigma_t = desired_sigma_t[i]

        for j in range(len(num_events)):

            event_positions, event_times = generate_events(num_events[j],
                                                           min_lat,
                                                           max_lat,
                                                           min_lon,
                                                           max_lon,
                                                           min_alt,
                                                           max_alt,
                                                           min_time,
                                                           max_time,
                                                           sigma_t=sigma_t)

            # gerando as detecções
            (detections,
             detection_times,
             n_event_positions,
             n_event_times,
             distances,
             spatial_clusters) = generate_detections(event_positions,
                                                     event_times,
                                                     sensors)

            # tudo pronto, rodando o runner

            (sol_centroides_espaciais,
             sol_centroides_temporais,
             sol_detectores,
             sol_best_fitness,
             sol_reference,
             delta_d,
             delta_t,
             execution_time,
             associacoes_corretas) = runner_ESO(event_positions,
                                                event_times,
                                                spatial_clusters,
                                                detections,
                                                detection_times,
                                                sensors,
                                                min_alt,
                                                max_alt,
                                                CLUSTER_MIN_PTS,
                                                sigma_t,
                                                AVG_LIGHT_SPEED * sigma_t,
                                                EPSILON_T,
                                                AVG_LIGHT_SPEED,
                                                False)
            print(delta_d)
            print(event_times)
            print(sol_centroides_temporais)
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
                prmse_temporal = calcula_prmse(
                    rmse_temporal, AVG_LIGHT_SPEED * sigma_t)
                mae_temporal = mae(delta_t)
                mle_temporal = np.abs(mean_location_error(delta_t))
                amse_temporal = average_mean_squared_error(delta_t)

                # dados espaciais

                # crlb_espacial_rmse = AVG_LIGHT_SPEED * crlb_temporal_rmse
                # crlb_espacial_medio = AVG_LIGHT_SPEED * crlb_temporal_medio
                # rmse_espacial = AVG_LIGHT_SPEED * rmse_temporal
                # prmse_espacial = AVG_LIGHT_SPEED * prmse_temporal
                # mae_espacial = AVG_LIGHT_SPEED * mae_temporal
                # mle_espacial = AVG_LIGHT_SPEED * mle_temporal
                # amse_espacial = AVG_LIGHT_SPEED * amse_temporal

                # dados espaciais
                crlb_espacial = ((AVG_LIGHT_SPEED * sigma_t) ** 2) / 7.0
                crlb_espacial_rmse = np.sqrt(crlb_espacial)
                crlb_espacial_medio = crlb_espacial
                rmse_espacial = rmse(delta_d)
                prmse_espacial = calcula_prmse(
                    rmse_espacial, AVG_LIGHT_SPEED * sigma_t)
                mae_espacial = mae(delta_d)
                mle_espacial = np.abs(mean_location_error(delta_d))
                amse_espacial = average_mean_squared_error(delta_d)

                # erro relativo

                relative_error = erro_relativo_funcao_ajuste(sol_best_fitness,
                                                             sol_reference)

                # porcentagem das associações corretas

                correct_association_index = np.mean(associacoes_corretas) * 100

                # adicionando no paper

                paper_data.append([sigma_t/SIGMA_T,
                                   float(crlb_espacial_medio),
                                   float(crlb_espacial_rmse),
                                   float(rmse_espacial),
                                   float(prmse_espacial),
                                   float(mae_espacial),
                                   float(mle_espacial),
                                   float(amse_espacial),
                                   float(crlb_temporal_medio),
                                   float(crlb_temporal_rmse),
                                   float(rmse_temporal),
                                   float(prmse_temporal),
                                   float(mae_temporal),
                                   float(mle_temporal),
                                   float(amse_temporal),
                                   float(relative_error),
                                   float(correct_association_index)
                                   ])

                print([sigma_t/SIGMA_T,
                       float(crlb_espacial_medio),
                       float(crlb_espacial_rmse),
                       float(rmse_espacial),
                       float(prmse_espacial),
                       float(mae_espacial),
                       float(mle_espacial),
                       float(amse_espacial),
                       float(crlb_temporal_medio),
                       float(crlb_temporal_rmse),
                       float(rmse_temporal),
                       float(prmse_temporal),
                       float(mae_temporal),
                       float(mle_temporal),
                       float(amse_temporal),
                       float(relative_error),
                       float(correct_association_index)])

    # salvando o arquivo
    if fake_test:
        """
        Salvando o arquivo para o paper
        """
        """
        Salvando o arquivo para o paper
        """
        paper_data = np.array(paper_data)
        output_file = os.path.join(basedir, "ESO_results.csv")
        np.savetxt(output_file,
                   paper_data,
                   delimiter=";",
                   fmt="%.5f",
                   header="SigmaT_ratio;CRLB_Spatial_Mean;CRLB_Spatial_RMSE;RMSE_Spatial;"
                          "PRMSE_Spatial;MAE_Spatial;MLE_Spatial;AMSE_Spatial;"
                          "CRLB_Temporal_Mean;CRLB_Temporal_RMSE;RMSE_Temporal;"
                          "PRMSE_Temporal;MAE_Temporal;MLE_Temporal;AMSE_Temporal;"
                          "Relative_Error;Correct_Association(%)",
                   comments='')
        print(f"\n>> Resultados salvos em: {output_file}")


if __name__ == "__main__":
    test_runner_ESO(True)
