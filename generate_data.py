
from GeoLightning.Simulator.Simulator import *
from GeoLightning.Utils.Utils import *
from GeoLightning.Utils.Constants import *

# recuperando o grupo de sensores
sensors = get_sensors()
min_lat, max_lat, min_lon, max_lon = get_lightning_limits(sensors)

# gerando os eventos
min_alt = 0
max_alt = 10
min_time = 10000
max_time = 11000
num_events = [2, 5, 10, 15, 20, 25,
              30, 100, 500, 800, 1000,
              2000, 3000, 4000, 5000, 6000,
              7000, 8000, 9000, 10000, 20000]

print(min_lat, max_lat, min_lon, max_lon, min_alt, max_alt)

for i in range(len(num_events)):

    file_detections = "data/static_constellation_detections_{:06d}.npy".format(
        num_events[i])

    file_detections_times = "data/static_constelation_detection_times_{:06d}.npy".format(
        num_events[i])

    file_event_positions = "data/static_constelation_event_positions_{:06d}.npy".format(
        num_events[i])

    file_event_times = "data/static_constelation_event_times_{:06d}.npy".format(
        num_events[i])

    file_n_event_positions = "data/static_constelation_n_event_positions_{:06d}.npy".format(
        num_events[i])

    file_n_event_times = "data/static_constelation_n_event_times_{:06d}.npy".format(
        num_events[i])

    file_distances = "data/static_constelation_distances_{:06d}.npy".format(
        num_events[i])

    file_spatial_clusters = "data/static_constelation_spatial_clusters_{:06d}.npy".format(
        num_events[i])

    event_positions, event_times = generate_events(num_events[i],
                                                   min_lat,
                                                   max_lat,
                                                   min_lon,
                                                   max_lon,
                                                   min_alt,
                                                   max_alt,
                                                   min_time,
                                                   max_time)

    np.save(file_event_positions, event_positions, allow_pickle=True)

    np.save(file_event_times, event_times, allow_pickle=True)

    # gerando as detecções
    (detections,
        detection_times,
        n_event_positions,
        n_event_times,
        distances,
        spatial_clusters) = generate_detections(event_positions,
                                                event_times,
                                                sensors)

    event_times = computa_tempos_de_origem(n_event_positions,
                                           spatial_clusters,
                                           detection_times,
                                           detections)

    print(np.allclose(event_times, n_event_times, rtol=3 * SIGMA_T))

    np.save(file_detections, detections, allow_pickle=True)
    np.save(file_detections_times, detection_times, allow_pickle=True)
    np.save(file_n_event_positions, n_event_positions, allow_pickle=True)
    np.save(file_n_event_times, n_event_times, allow_pickle=True)
    np.save(file_distances, distances, allow_pickle=True)
    np.save(file_spatial_clusters, spatial_clusters, allow_pickle=True)
