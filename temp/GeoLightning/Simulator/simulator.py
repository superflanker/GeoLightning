import numpy as np
from GeoLightning.Utils.haversine import haversine

def generate_sensors(num_sensors=5, lat_range=(-10, -5), lon_range=(-50, -45)):
    lats = np.random.uniform(lat_range[0], lat_range[1], num_sensors)
    lons = np.random.uniform(lon_range[0], lon_range[1], num_sensors)
    return np.column_stack((lats, lons))

def generate_events(num_events=3, lat_range=(-10, -5), lon_range=(-50, -45), time_range=(0, 10)):
    lats = np.random.uniform(lat_range[0], lat_range[1], num_events)
    lons = np.random.uniform(lon_range[0], lon_range[1], num_events)
    times = np.random.uniform(time_range[0], time_range[1], num_events)
    return np.column_stack((lats, lons, times))

def generate_detections(events, sensors, speed=299_792.458, jitter_std=0.0001):
    num_events = events.shape[0]
    num_sensors = sensors.shape[0]

    detections = []
    for event_id in range(num_events):
        lat_ev, lon_ev, t_ev = events[event_id]
        for sensor_id in range(num_sensors):
            lat_s, lon_s = sensors[sensor_id]
            distance = haversine(lat_ev, lon_ev, lat_s, lon_s) * 1000.0
            propagation_time = distance / speed
            timestamp = t_ev + propagation_time + np.random.normal(0, jitter_std)
            detections.append((event_id, sensor_id, timestamp))

    return np.array(detections)