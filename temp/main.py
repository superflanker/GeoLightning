from GeoLightning.Simulator.simulator import generate_sensors, generate_events, generate_detections
from GeoLightning.Optimization.genetic_solver import run_ga_numba

sensors = generate_sensors(5)
events = generate_events(3)
detections = generate_detections(events, sensors)

best_solution, best_fitness = run_ga_numba(detections, sensors, num_events=5)

print("Melhor solução:", best_solution)
print("Fitness:", best_fitness)