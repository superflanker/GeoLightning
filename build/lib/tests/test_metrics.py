"""
EELT 7019 - Applied Artificial Intelligence
Metrics Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Simulator.Metrics import (
    rmse,
    mae,
    average_mean_squared_error,
    mean_location_error,
    calcula_prmse,
    acuracia_associacao,
    erro_relativo_funcao_ajuste,
    tempo_execucao
)

def test_rmse():
    delta = np.random.rand(10)
    resultado = rmse(delta)
    esperado = np.sqrt(np.mean((delta) ** 2))
    assert np.isclose(resultado, esperado)

def test_mae():
    delta = np.random.rand(10)
    resultado = mae(delta)
    esperado = np.mean(np.abs(delta))
    assert np.isclose(resultado, esperado)

def test_average_mean_squared_error():
    
    delta = np.random.rand(10)
    resultado = average_mean_squared_error(delta)
    esperado = np.mean(delta ** 2)
    assert np.isclose(resultado, esperado)

def test_mean_location_error():
    delta = np.random.rand(10)
    resultado = mean_location_error(delta)
    esperado = np.mean(delta)
    assert np.isclose(resultado, esperado)

def test_calcula_prmse():
    resultado = calcula_prmse(10.0, 100.0)
    assert np.isclose(resultado, 10.0)

def test_acuracia_associacao():
    reais = np.array([0, 1, 2, 3])
    estimadas = np.array([0, 1, 0, 3])
    resultado = acuracia_associacao(estimadas, reais)
    assert np.isclose(resultado, 0.75)

def test_erro_relativo_funcao_ajuste():
    resultado = erro_relativo_funcao_ajuste(90.0, 100.0)
    assert np.isclose(resultado, 10.0)

def test_tempo_execucao():
    assert np.isclose(tempo_execucao(2.0, 5.0), 3.0)


if __name__ == "__main__":
    test_rmse()
    test_mae()
    test_average_mean_squared_error()
    test_mean_location_error()
    test_calcula_prmse()
    test_acuracia_associacao()
    test_erro_relativo_funcao_ajuste()
    test_tempo_execucao()
