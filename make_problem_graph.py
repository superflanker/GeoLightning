"""
EELT 7019 - Applied Artificial Intelligence
Problem Plot - STELA Theory
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
# configuração do matplotlib
# plt.style.use(['science'])
# Coordenadas dos sensores em disposição hexagonal com 1 no centro
radius = 80  # raio do hexágono (km)
angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
sensor_positions = [(0, 0)]  # sensor central
sensor_positions += [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
sensor_positions = np.array(sensor_positions)

# Dois eventos aleatórios dentro da área central
# np.random.seed(42)
event_positions = np.random.uniform(-50, 50, size=(2, 2))

# Define cores
sensor_color = 'blue'
event_color = 'green'
circle_color = 'black'

# Cria a figura
fig, ax = plt.subplots(figsize=(6, 6))

# Plota os sensores
ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1],
           c=sensor_color, s=60, label='Sensors')

# Plota os eventos
ax.scatter(event_positions[:, 0], event_positions[:, 1],
           c=event_color, s=60, label='Events')

# Plota bandas circulares para 3 sensores por evento
for i, event in enumerate(event_positions):
    for j in [0, 1, 2, 3, 4, 5, 6]:  # usa três sensores não consecutivos
        d = np.linalg.norm(sensor_positions[j] - event)
        circle = plt.Circle(sensor_positions[j], d,
                            color=event_color, fill=False, linestyle='--', linewidth=1)
        ax.add_patch(circle)

# Ajustes de estilo IEEE
ax.set_aspect('equal')
ax.set_xlabel('X Position (km)', fontsize=10)
ax.set_ylabel('Y Position (km)', fontsize=10)
ax.grid(True, linestyle=':', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("data/images/ieee_hex_sensor_constellation.png", dpi=300)
plt.show()
