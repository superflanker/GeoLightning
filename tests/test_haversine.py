"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes Unitários - Função haversine - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from GeoLightning.Utils.utils import haversine

def test_haversine_zero_distance():
    assert haversine(0, 0, 0, 0) == 0

def test_haversine_known_distance():
    dist = haversine(-23.5505, -46.6333, -22.9068, -43.1729)
    print(dist)
    assert abs(dist - 360) < 1