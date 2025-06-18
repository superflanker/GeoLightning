"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes de Métricas
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
import pytest
from GeoLightning.Simulator.Metrics import (
    rmse,
    mae,
    average_mean_squared_error,
    mean_location_error,
    calcula_prmse,
    acuracia_associacao,
    erro_relativo_funcao_ajuste,
    tempo_execucao,
    calcular_crlb_espacial,
    calcular_crlb_temporal,
    calcular_crlb_rmse,
    calcular_mean_crlb
)

def test_rmse():
    reais = np.array([[0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0]])
    estimadas = np.array([[1.0, 0.0, 0.0],
                          [1.0, 2.0, 1.0]])
    resultado = rmse(estimadas, reais)
    esperado = np.sqrt(((1.0**2 + 0 + 0) + (0 + 1.0**2 + 0)) / 2)
    assert np.isclose(resultado, esperado)

def test_mae():
    reais = np.array([1.0, 2.0, 3.0])
    estimados = np.array([2.0, 2.0, 1.0])
    resultado = mae(estimados, reais)
    esperado = np.mean(np.abs(estimados - reais))
    assert np.isclose(resultado, esperado)

def test_average_mean_squared_error():
    reais = np.array([[0, 0, 0], [1, 1, 1]])
    estimados = np.array([[1, 1, 1], [1, 1, 1]])
    resultado = average_mean_squared_error(estimados, reais)
    esperado = (3 + 0) / 2  # erro quadrático médio
    assert np.isclose(resultado, esperado)

def test_mean_location_error():
    reais = np.array([[0, 0, 0], [0, 0, 0]])
    estimados = np.array([[1, 0, 0], [0, 3, 0]])
    resultado = mean_location_error(estimados, reais)
    esperado = (1 + 3) / 2
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

def test_crlb_espacial():
    crlb = calcular_crlb_espacial(sigma_d=1.0, N=4)
    esperado = (1.0 ** 2 / 4.0) * np.eye(3)
    assert np.allclose(crlb, esperado)

def test_crlb_temporal():
    crlb = calcular_crlb_temporal(sigma_t=0.5, N=4)
    esperado = (0.5 ** 2 / 4.0) * np.eye(1)
    assert np.allclose(crlb, esperado)

def test_crlb_rmse():
    crlb = np.diag([1.0, 1.0, 1.0])
    resultado = calcular_crlb_rmse(crlb)
    esperado = np.sqrt(np.trace(crlb @ crlb)) / 3.0
    assert np.isclose(resultado, esperado)

def test_mean_crlb():
    crlb = np.diag([1.0, 2.0, 3.0])
    resultado = calcular_mean_crlb(crlb)
    esperado = (1 + 2 + 3) / 3
    assert np.isclose(resultado, esperado)
