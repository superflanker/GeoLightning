"""
    EELT 7019 - Inteligência Artificial Aplicada
    Testes Unitários - Funçóes Gerados de Eventos e Sensores - Geolocalização de eventos atmosféricos
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

from GeoLightning.Simulator.simulator import generate_sensors, generate_events

def test_generate_sensors_correct_length():
    """Verifica se a quantidade de sensores gerada é correta."""
    sensors = generate_sensors(num_sensors=5)
    assert len(sensors) == 5, "Deveria gerar exatamente 5 sensores."


def test_generate_sensors_structure():
    """Verifica se os sensores têm a estrutura correta (lat, lon)."""
    sensors = generate_sensors(num_sensors=3)
    for sensor in sensors:
        assert isinstance(sensor, tuple), "Cada sensor deve ser uma tupla."
        assert len(sensor) == 2, "Cada sensor deve conter latitude e longitude."
        assert all(isinstance(coord, float) for coord in sensor), "Lat e Lon devem ser floats."


def test_generate_events_correct_length():
    """Verifica se a quantidade de eventos gerada é correta."""
    events = generate_events(num_events=4)
    assert len(events) == 4, "Deveria gerar exatamente 4 eventos."


def test_generate_events_structure():
    """Verifica se os eventos têm a estrutura correta (lat, lon, tempo)."""
    events = generate_events(num_events=2)
    for event in events:
        assert isinstance(event, tuple), "Cada evento deve ser uma tupla."
        assert len(event) == 3, "Cada evento deve conter latitude, longitude e tempo."
        assert all(isinstance(value, float) for value in event), "Todos os valores devem ser floats."


def test_generate_sensors_range():
    """Verifica se os sensores estão dentro do intervalo especificado."""
    lat_range = (-15, -10)
    lon_range = (-60, -50)
    sensors = generate_sensors(num_sensors=10, lat_range=lat_range, lon_range=lon_range)
    for lat, lon in sensors:
        assert lat_range[0] <= lat <= lat_range[1], "Latitude fora do intervalo."
        assert lon_range[0] <= lon <= lon_range[1], "Longitude fora do intervalo."


def test_generate_events_range():
    """Verifica se os eventos estão dentro dos intervalos especificados."""
    lat_range = (-20, -15)
    lon_range = (-60, -55)
    time_range = (0, 100)

    events = generate_events(
        num_events=5, lat_range=lat_range, lon_range=lon_range, time_range=time_range
    )

    for lat, lon, t in events:
        assert lat_range[0] <= lat <= lat_range[1], "Latitude fora do intervalo."
        assert lon_range[0] <= lon <= lon_range[1], "Longitude fora do intervalo."
        assert time_range[0] <= t <= time_range[1], "Tempo fora do intervalo."
