"""
EELT 7019 - Applied Artificial Intelligence
Sensor Simulation Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from GeoLightning.GraphUtils.Graphics import make_histogram_graph
from GeoLightning.Utils.Constants import SIGMA_D, SIGMA_T, AVG_LIGHT_SPEED
from matplotlib.colors import LinearSegmentedColormap


def compute_90_percentile(hist,
                          bin_edges,
                          total):
    begin_range = bin_edges[0]
    end_range = bin_edges[-1]
    rsum = 0
    for ix in range(0, len(hist)):
        rsum += hist[ix]
        if rsum >= 0.9 * total:
            end_range = bin_edges[ix]
            break
    return begin_range, end_range


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


calibration_files = ["data/LSA_calibration.npy",
                     "data/ESO_calibration.npy"]

final_results_files = ["data/LSA_final_results.npy",
                       "data/ESO_final_results.npy"]

calibration_table = [["LSA"],
                     ["ESO"]]

# tabelas

tuning_table = [["PSO", 400, 80],
                ["GA", 200, 100],
                ["GWO", 300, 100],
                ["AOA", 1000, 50],
                ["LSA", 150, 40],
                ["FHO", 450, 70],
                ["ESO", 150, 30]]

final_tuning_table = [["LSA", 6, 11],
                      ["ESO", 15, 16]]

results = list()

for i in range(len(results_files)):
    result_file = results_files[i]
    t_results = np.load(result_file)
    results.append(tuning_table[i] + t_results.tolist())

calibration_results = list()

for i in range(len(calibration_files)):
    result_file = calibration_files[i]
    t_results = np.load(result_file)
    for j in range(len(t_results)):
        calibration_results.append(
            calibration_table[i] + t_results[j].tolist())

final_results = list()

for i in range(len(final_results_files)):
    result_file = final_results_files[i]
    t_results = np.load(result_file)
    final_results.append(final_tuning_table[i] + t_results.tolist())

df = pd.DataFrame(results, columns=[r"Algorithm",
                                    r"Epochs",
                                    r"Population",
                                    r"Sigma Ratio",
                                    r"CRLB_{e}",
                                    r"RMSE_{e}",
                                    r"RMSE_{e_{improved}}",
                                    r"PRMSE_{e}",
                                    r"PRMSE_{e_{improved}}",
                                    r"MLE_{e}",
                                    r"MLE_{e_{improved}}",
                                    r"MAE_{e}",
                                    r"MAE_{e_{improved}}",
                                    r"AMSE_{e}",
                                    r"AMSE_{e_{improved}}",
                                    r"CRLB_{t}",
                                    r"RMSE_{t}",
                                    r"RMSE_{t_{improved}}",
                                    r"MLE_{t}",
                                    r"MLE_{t_{improved}}",
                                    r"MAE_{t}",
                                    r"MAE_{t_{improved}}",
                                    r"AMSE_{t}",
                                    r"AMSE_{t_{improved}}",
                                    r"CAT",
                                    r"Best Value",
                                    r"Best Value (improved)",
                                    r"Reference Value",
                                    r"Execution Time (s)"])

df[r"V_{ratio}"] = 100 * \
    (abs(df["Best Value"] - df["Reference Value"])) / df["Reference Value"]

df_tuning = df[[r"Algorithm",
                r"Epochs",
                r"Population",
                r"Best Value",
                r"Reference Value",
                r"V_{ratio}",
                r"Execution Time (s)"]].copy()

df_comparision = df[[r"Algorithm",
                     r"CRLB_{e}",
                     r"RMSE_{e}",
                     r"PRMSE_{e}",
                     r"MLE_{e}",
                     r"MLE_{t}"]].copy()

df_calibration = pd.DataFrame(calibration_results, columns=[r"Algorithm",
                                                            r"Epochs",
                                                            r"Population",
                                                            r"Sigma Ratio",
                                                            r"CRLB_{e}",
                                                            r"RMSE_{e}",
                                                            r"RMSE_{e_{improved}}",
                                                            r"PRMSE_{e}",
                                                            r"PRMSE_{e_{improved}}",
                                                            r"MLE_{e}",
                                                            r"MLE_{e_{improved}}",
                                                            r"MAE_{e}",
                                                            r"MAE_{e_{improved}}",
                                                            r"AMSE_{e}",
                                                            r"AMSE_{e_{improved}}",
                                                            r"CRLB_{t}",
                                                            r"RMSE_{t}",
                                                            r"RMSE_{t_{improved}}",
                                                            r"MLE_{t}",
                                                            r"MLE_{t_{improved}}",
                                                            r"MAE_{t}",
                                                            r"MAE_{t_{improved}}",
                                                            r"AMSE_{t}",
                                                            r"AMSE_{t_{improved}}",
                                                            r"CAT",
                                                            r"Best Value",
                                                            r"Best Value (improved)",
                                                            r"Reference Value",
                                                            r"Execution Time (s)"])


df_final = pd.DataFrame(final_results, columns=[r"Algorithm",
                                                r"Epochs",
                                                r"Population",
                                                r"Sigma Ratio",
                                                r"CRLB_{e}",
                                                r"RMSE_{e}",
                                                r"RMSE_{e_{improved}}",
                                                r"PRMSE_{e}",
                                                r"PRMSE_{e_{improved}}",
                                                r"MLE_{e}",
                                                r"MLE_{e_{improved}}",
                                                r"MAE_{e}",
                                                r"MAE_{e_{improved}}",
                                                r"AMSE_{e}",
                                                r"AMSE_{e_{improved}}",
                                                r"CRLB_{t}",
                                                r"RMSE_{t}",
                                                r"RMSE_{t_{improved}}",
                                                r"MLE_{t}",
                                                r"MLE_{t_{improved}}",
                                                r"MAE_{t}",
                                                r"MAE_{t_{improved}}",
                                                r"AMSE_{t}",
                                                r"AMSE_{t_{improved}}",
                                                r"CAT",
                                                r"Best Value",
                                                r"Best Value (improved)",
                                                r"Reference Value",
                                                r"Execution Time (s)"])

df_final[r"V_{ratio}"] = 100 * \
    (abs(df_final["Best Value"] - df_final["Reference Value"])) / \
    df_final["Reference Value"]

df_final_tuning = df_final[[r"Algorithm",
                            r"Epochs",
                            r"Population",
                            r"Best Value",
                            r"Reference Value",
                            r"V_{ratio}",
                            r"Execution Time (s)"]].copy()

df_final_comparision = df_final[[r"Algorithm",
                                 r"CRLB_{e}",
                                 r"RMSE_{e}",
                                 r"MLE_{e}",
                                 r"MLE_{t}",
                                 r"RMSE_{e_{improved}}",
                                 r"MLE_{e_{improved}}",
                                 r"MLE_{t_{improved}}"]].copy()

df_comparision[r"MLE_{t}"] = df_comparision[r"MLE_{t}"].apply(
    lambda x: f"{x:.2e}")

df_calibration[r"MLE_{t}"] = df_calibration[r"MLE_{t}"].apply(
    lambda x: f"{x:.2e}")

df_calibration[r"MLE_{t_{improved}}"] = df_calibration[r"MLE_{t_{improved}}"].apply(
    lambda x: f"{x:.2e}")

df_final_comparision[r"MLE_{t}"] = df_final_comparision[r"MLE_{t}"].apply(
    lambda x: f"{x:.2e}")

df_final_comparision[r"MLE_{t_{improved}}"] = df_final_comparision[r"MLE_{t_{improved}}"].apply(
    lambda x: f"{x:.2e}")

print(df_tuning.to_latex(index=False, float_format="%.2f"))

print(df_comparision.to_latex(index=False, float_format="%.2f"))

print(df_final_tuning.to_latex(index=False, float_format="%.2f"))

print(df_final_comparision.to_latex(index=False, float_format="%.2f"))

#############################################
# LSA - Erro em função da época e população #
#############################################
meu_cinza = LinearSegmentedColormap.from_list("custom_gray", ["0.1", "0.7"])

df_lsa = df_calibration[df_calibration["Algorithm"] == "LSA"]

plt.close('all')

x = df_lsa["Epochs"].values
y = df_lsa["Population"].values
z = df_lsa["MLE_{e}"].values
z_mle = df_lsa["MLE_{e}"].values

# 2. Criação de uma grade regular e densa para suavização
xi = np.linspace(x.min(), x.max(), 100)  # 100 pontos de resolução
yi = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(xi, yi)

# 2. Coletar os pontos (x, y) da curva de nível
# Usamos a versão 2D para extrair as coordenadas sem interferência do 3D
fig_temp = plt.figure()  # Figura temporária para não sujar o gráfico atual
contorno_mle = plt.tricontour(x, y, z_mle, levels=[3 * SIGMA_D])

pontos_da_linha = []
# A forma mais compatível de acessar os caminhos:
for path in contorno_mle.get_paths():
    v = path.vertices
    pontos_da_linha.append(v)

plt.close(fig_temp)  # Fecha a figura temporária

if not pontos_da_linha:
    raise ValueError(
        "A curva de nível 3*SIGMA_D não foi encontrada nos dados.")

# Unifica todos os segmentos
pontos_xy_isolinha = np.vstack(pontos_da_linha)

z_time = df_lsa["Execution Time (s)"].values
Z_time = griddata((x, y), z_time, (X, Y), method='cubic')

# 3. Interpolar o Tempo de Execução sobre esses pontos específicos
# Usamos o griddata novamente, mas agora apenas para os pontos da linha
tempos_na_linha = griddata(
    (x, y), z_time, (pontos_xy_isolinha[:, 0], pontos_xy_isolinha[:, 1]), method='cubic')

erros_na_linha = griddata(
    (x, y), z_mle, (pontos_xy_isolinha[:, 0], pontos_xy_isolinha[:, 1]), method='cubic')

# 4. Criar um DataFrame para análise
df_otimizacao = pd.DataFrame({
    'Epochs': pontos_xy_isolinha[:, 0],
    'Population': pontos_xy_isolinha[:, 1],
    'Execution_Time': tempos_na_linha,
    'MLE': erros_na_linha
})

melhor_config = df_otimizacao.loc[df_otimizacao['Execution_Time'].idxmin()]

plt.close('all')
z_mle = df_lsa["MLE_{e}"].values
Z_mle = griddata((x, y), z_mle, (X, Y), method='cubic')

fig1 = plt.figure(figsize=(12, 7))
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_mle, cmap="coolwarm",
                         alpha=0.6, antialiased=True)

ax1.plot(
    df_otimizacao['Epochs'],
    df_otimizacao['Population'],
    df_otimizacao['MLE'],
    color='#282b28',
    linewidth=2,
    linestyle="-.",
    label='$3\sigma_d$ frontier',
    zorder=10
)

ax1.scatter(
    melhor_config['Epochs'],
    melhor_config['Population'],
    melhor_config['MLE'],
    color='#494d49',
    s=100,
    edgecolor='#494d49',
    marker='*',
    zorder=20,
    label='Optimal point'
)

ax1.legend(loc='upper left', fontsize=10)

ax1.xaxis._axinfo["grid"]['linewidth'] = 0.5
ax1.yaxis._axinfo["grid"]['linewidth'] = 0.5
ax1.zaxis._axinfo["grid"]['linewidth'] = 0.5

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Population')
ax1.set_zlabel('$MLE_{e}$')
ax1.view_init(elev=30, azim=60)
plt.savefig("data/images/grafico_calibracao_MLE_LSA.png",
            dpi=600, bbox_inches="tight", pad_inches=0.5)

##########################################################
# LSA - Tempo de Execução em função da época e população #
##########################################################

fig2 = plt.figure(figsize=(12, 7))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_time, cmap="coolwarm",
                         alpha=0.6, antialiased=True)

# cbar2 = fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
# cbar2.set_label('Tempo (s)', fontsize=12)

print(f"\n--- OTIMIZAÇÃO ENCONTRADA (LSA) ---")
print(f"Para manter o erro MLE em {3 * SIGMA_D} m:")
print(f"Melhor Epochs: {int(np.ceil(melhor_config['Epochs']))}")
print(f"Melhor Population: {int(np.ceil(melhor_config['Population']))}")
print(f"Tempo Médio de Execução: {melhor_config['Execution_Time']:.4f}s")

ax2.plot(
    df_otimizacao['Epochs'],
    df_otimizacao['Population'],
    df_otimizacao['Execution_Time'],
    color='#282b28',
    linewidth=2,
    linestyle="-.",
    label='$3\sigma_d$ frontier',
    zorder=10
)

ax2.scatter(
    melhor_config['Epochs'],
    melhor_config['Population'],
    melhor_config['Execution_Time'],
    color='#494d49',
    s=100,
    edgecolor='#282b28',
    marker='*',
    label='Optimal point',
    zorder=20
)

ax2.legend(loc='upper left', fontsize=10)

ax2.xaxis._axinfo["grid"]['linewidth'] = 0.5
ax2.yaxis._axinfo["grid"]['linewidth'] = 0.5
ax2.zaxis._axinfo["grid"]['linewidth'] = 0.5

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Population')
ax2.set_zlabel('Execution Time (s)')
ax2.view_init(elev=30, azim=-150)
plt.savefig("data/images/grafico_calibracao_EXECUÇÂO_LSA.png",
            dpi=600, bbox_inches="tight", pad_inches=0.5)

#############################################
# ESO - Erro em função da época e população #
#############################################

df_ESO = df_calibration[df_calibration["Algorithm"] == "ESO"]

plt.close('all')

x = df_ESO["Epochs"].values
y = df_ESO["Population"].values
z = df_ESO["MLE_{e}"].values
z_mle = df_ESO["MLE_{e}"].values

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
X, Y = np.meshgrid(xi, yi)

fig_temp = plt.figure()
contorno_mle = plt.tricontour(x, y, z_mle, levels=[3 * SIGMA_D])

pontos_da_linha = []

for path in contorno_mle.get_paths():
    v = path.vertices
    pontos_da_linha.append(v)

plt.close(fig_temp)

if not pontos_da_linha:
    raise ValueError(
        "A curva de nível 3*SIGMA_D não foi encontrada nos dados.")

pontos_xy_isolinha = np.vstack(pontos_da_linha)

z_time = df_ESO["Execution Time (s)"].values
Z_time = griddata((x, y), z_time, (X, Y), method='cubic')

tempos_na_linha = griddata(
    (x, y), z_time, (pontos_xy_isolinha[:, 0], pontos_xy_isolinha[:, 1]), method='cubic')

erros_na_linha = griddata(
    (x, y), z_mle, (pontos_xy_isolinha[:, 0], pontos_xy_isolinha[:, 1]), method='cubic')

df_otimizacao = pd.DataFrame({
    'Epochs': pontos_xy_isolinha[:, 0],
    'Population': pontos_xy_isolinha[:, 1],
    'Execution_Time': tempos_na_linha,
    'MLE': erros_na_linha
})

# 5. Encontrar o ponto de Tempo Mínimo
melhor_config = df_otimizacao.loc[df_otimizacao['Execution_Time'].idxmin()]

# --- ESO - Erro em função da época e população ---
plt.close('all')
z_mle = df_ESO["MLE_{e}"].values
Z_mle = griddata((x, y), z_mle, (X, Y), method='cubic')

fig1 = plt.figure(figsize=(12, 7))
ax1 = fig1.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z_mle, cmap="coolwarm",
                         alpha=0.6, antialiased=True)

# cbar1 = fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10, pad=0.1)
# cbar1.set_label('$MLE_{e}$', fontsize=12)

ax1.plot(
    df_otimizacao['Epochs'],
    df_otimizacao['Population'],
    df_otimizacao['MLE'],
    color='#282b28',
    linewidth=2,
    linestyle="-.",
    label='$3\sigma_d$ frontier',
    zorder=10
)

ax1.scatter(
    melhor_config['Epochs'],
    melhor_config['Population'],
    melhor_config['MLE'],
    color='#494d49',
    s=100,
    edgecolor='#282b28',
    marker='*',
    zorder=20,
    label='Optimal point'
)


ax1.xaxis._axinfo["grid"]['linewidth'] = 0.5
ax1.yaxis._axinfo["grid"]['linewidth'] = 0.5
ax1.zaxis._axinfo["grid"]['linewidth'] = 0.5

ax1.legend(loc='upper left', fontsize=10)

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Population')
ax1.set_zlabel('$MLE_{e}$')
ax1.view_init(elev=30, azim=60)
plt.savefig("data/images/grafico_calibracao_MLE_ESO.png",
            dpi=600, bbox_inches="tight", pad_inches=0.5)

##########################################################
# ESO - Tempo de Execução em função da época e população #
##########################################################

z_time = df_ESO["Execution Time (s)"].values
Z_time = griddata((x, y), z_time, (X, Y), method='cubic')

fig2 = plt.figure(figsize=(12, 7))
ax2 = fig2.add_subplot(111, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z_time, cmap="coolwarm",
                         alpha=0.6, antialiased=True)

# cbar2 = fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10, pad=0.1)
# cbar2.set_label('Tempo (s)', fontsize=12)

print(f"\n--- OTIMIZAÇÃO ENCONTRADA (ESO) ---")
print(f"Para manter o erro MLE em {3 * SIGMA_D} m:")
print(f"Melhor Epochs: {int(np.ceil(melhor_config['Epochs']))}")
print(f"Melhor Population: {int(np.ceil(melhor_config['Population']))}")
print(f"Tempo Médio de Execução: {melhor_config['Execution_Time']:.4f}s")

ax2.plot(
    df_otimizacao['Epochs'],
    df_otimizacao['Population'],
    df_otimizacao['Execution_Time'],
    color='#282b28',
    linewidth=2,
    linestyle="-.",
    label='$3\sigma_d$ frontier',
    zorder=10
)

ax2.scatter(
    melhor_config['Epochs'],
    melhor_config['Population'],
    melhor_config['Execution_Time'],
    color='#494d49',
    s=100,
    edgecolor='#282b28',
    marker='*',
    label='Optimal point',
    zorder=20
)

ax2.legend(loc='upper left', fontsize=10)

ax2.xaxis._axinfo["grid"]['linewidth'] = 0.5
ax2.yaxis._axinfo["grid"]['linewidth'] = 0.5
ax2.zaxis._axinfo["grid"]['linewidth'] = 0.5

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Population')
ax2.set_zlabel('Execution Time (s)')
ax2.view_init(elev=30, azim=-150)
plt.tight_layout()
plt.savefig("data/images/grafico_calibracao_EXECUÇÂO_ESO.png",
            dpi=600, bbox_inches="tight", pad_inches=0.5)


bins = 200

# erros de localização e temporais

delta_d_files = ["data/LSA_final_deltas_distancia_refinado.npy",
                 "data/ESO_final_deltas_distancia_refinado.npy"]

for delta_file in delta_d_files:
    data = np.load(delta_file).flatten()
    delta_file = delta_file.replace("data/", "data/images/")
    delta_file = delta_file.replace("npy", "png")
    print(delta_file)
    hist_values, bin_edges = np.histogram(data, bins=bins)
    b_size = 0.8 * (max(data) - min(data)) / bins

    total = np.sum(hist_values)
    quantile_90 = np.quantile(data, 0.9)
    hist_values = [x / total for x in hist_values]
    print(quantile_90)
    make_histogram_graph(hist=hist_values,
                         bin_edges=bin_edges,
                         quantile_90=quantile_90,
                         xlabel=r"Location Error ($m$)",
                         ylabel="Probability Density Function (PDF)",
                         xlimit=3500,
                         b_size=b_size,
                         filename=delta_file)

bins = 250

delta_t_files = ["data/LSA_final_deltas_tempos_refinado.npy",
                 "data/ESO_final_deltas_tempos_refinado.npy"]

for delta_file in delta_t_files:
    data = np.abs(np.load(delta_file)).flatten()
    delta_file = delta_file.replace("data/", "data/images/")
    delta_file = delta_file.replace("npy", "png")
    print(delta_file)
    hist_values, bin_edges = np.histogram(data, bins=bins)
    b_size = 0.8 * (max(data) - min(data)) / bins

    total = np.sum(hist_values)
    quantile_90 = np.quantile(data, 0.9)
    hist_values = [x / total for x in hist_values]
    print(quantile_90)
    make_histogram_graph(hist=hist_values,
                         bin_edges=bin_edges,
                         quantile_90=quantile_90,
                         xlabel=r"Temporal Error($s$)",
                         ylabel="Probability Density Function (PDF)",
                         xlimit=8e-6,
                         b_size=b_size,
                         filename=delta_file)
