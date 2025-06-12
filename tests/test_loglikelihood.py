
import numpy as np
from GeoLightning.Stela.LogLikelihood import funcao_log_verossimilhanca

def test_log_verossimilhanca_valores_normais():
    deltas = np.array([0.0, 0.1, -0.1, 0.2, -0.2])
    sigma = 0.1
    ll = funcao_log_verossimilhanca(deltas, sigma)
    assert isinstance(ll, float)

def test_log_verossimilhanca_zero_deltas():
    deltas = np.zeros(10)
    sigma = 1.0
    ll = funcao_log_verossimilhanca(deltas, sigma)
    assert np.isclose(ll, -0.5 * 10 * np.log(2 * np.pi * sigma ** 2))
