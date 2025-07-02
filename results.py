"""
EELT 7019 - Applied Artificial Intelligence
Sensor Simulation Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
import pandas as pd
from GeoLightning.GraphUtils.Graphics import make_histogram_graph


def round_values(value: np.float64) -> str:
    rounded_value = f"{value:.02f}"
    return rounded_value


# arquivos

results_files = ["data/PSO_results.npy",
                 "data/GA_results.npy",
                 "data/GWO_results.npy",
                 "data/AOA_results.npy",
                 "data/LSA_results.npy",
                 "data/FHO_results.npy",
                 "data/ESO_results.npy"]

# tabelas

tuning_table = [["PSO", 400, 80],
                ["GA", 200, 100],
                ["GWO", 300, 100],
                ["AOA", 1000, 50],
                ["LSA", 150, 40],
                ["FHO", 450, 70],
                ["ESO", 150, 30]]

results = list()

for i in range(len(results_files)):
    result_file = results_files[i]
    t_results = np.load(result_file)
    results.append(tuning_table[i] + t_results.tolist())

df = pd.DataFrame(results, columns=["Algorithm",
                                    "Epochs",
                                    "Population",
                                    "Sigma_Ratio",
                                    "CRLB_e",
                                    "RMSE_e",
                                    "PRMSE_e",
                                    "MLE_e",
                                    "AMSE_e",
                                    "CRLB_t",
                                    "RMSE_t",
                                    "MLE_t",
                                    "AMSE_t",
                                    "CAT",
                                    "Best Value",
                                    "Reference Value",
                                    "Execution Time (s)"])

df["V_{ratio}"] = 100 * \
    (abs(df["Best Value"] - df["Reference Value"])) / df["Reference Value"]

df_tuning = df[["Algorithm",
                "Epochs",
                "Population",
                "Best Value",
                "Reference Value",
                "V_{ratio}",
                "Execution Time (s)"]].copy()

df_comparision = df[["Algorithm",
                     "CRLB_e",
                     "RMSE_e",
                     "PRMSE_e",
                     "MLE_e",
                     "MLE_t"]].copy()

df_comparision["MLE_t"] = df_comparision["MLE_t"].apply(lambda x: f"{x:.2e}")

print(df_tuning.to_latex(index=False, float_format="%.2f"))

print(df_comparision.to_latex(index=False, float_format="%.2f"))

