from numba import njit
import numpy as np
from GeoLightning.Utils.haversine import haversine

@njit
def fitness_numba(solution, detections, sensors, max_events, speed=299_792.458):
    event_params = solution[:max_events * 4]
    assignments = solution[max_events * 4:].astype(np.int64)
    total_error = 0.0
    active_events = 0

    for idx in range(max_events):
        flag = event_params[idx * 4 + 3]
        if flag >= 0.5:
            active_events += 1

    for i in range(detections.shape[0]):
        event_idx = assignments[i]
        flag = event_params[event_idx * 4 + 3]
        if flag < 0.5:
            total_error += 10.0
            continue

        lat_ev = event_params[event_idx * 4]
        lon_ev = event_params[event_idx * 4 + 1]
        time_ev = event_params[event_idx * 4 + 2]

        sensor_id = int(detections[i, 1])
        timestamp = detections[i, 2]

        lat_s, lon_s = sensors[sensor_id]

        distance = haversine(lat_ev, lon_ev, lat_s, lon_s) * 1000.0
        expected_time = time_ev + (distance / speed)

        error = abs(timestamp - expected_time)
        total_error += error

    total_error += active_events * 0.5
    return total_error