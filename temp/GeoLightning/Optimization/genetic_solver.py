from mealpy.evolutionary_based import GA
from GeoLightning.Optimization.fitness import fitness_numba

def run_ga_numba(detections, sensors, num_events, epochs=100, pop_size=50):
    num_detections = detections.shape[0]

    lb = [-10, -50, 0, 0] * num_events + [0] * num_detections
    ub = [-5, -45, 10, 1] * num_events + [num_events - 1] * num_detections

    def problem_f(solution):
        return fitness_numba(solution, detections, sensors, num_events)

    problem = {
        "fit_func": problem_f,
        "lb": lb,
        "ub": ub,
        "minmax": "min",
    }

    model = GA.BaseGA(epoch=epochs, pop_size=pop_size)
    best_pos, best_fit = model.solve(problem)
    return best_pos, best_fit