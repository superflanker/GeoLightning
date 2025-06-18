"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes Entropia
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import numpy as np
from GeoLightning.Stela.Entropy import calcular_entropia_local

def test_entropia_com_valores_uniformes():
    tempos = np.linspace(0, 1, 100)
    entropia = calcular_entropia_local(tempos)
    assert entropia > 0

def test_entropia_com_valores_constantes():
    tempos = np.ones(100)
    entropia = calcular_entropia_local(tempos)
    assert entropia == 0.0

def test_entropia_com_um_valor():
    tempos = np.array([1.0])
    entropia = calcular_entropia_local(tempos)
    assert entropia == 0.0
