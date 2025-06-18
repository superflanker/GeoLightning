import sympy as sp

x, y, z, N = sp.symbols("x, y, z, N")

x_s, y_s, z_s = sp.symbols("x_s, y_s, z_s")

s_d = sp.symbols("sigma_d")

print(x, y, z, s_d)

# Função log-verossimilhança negativa (para FIM)
l = ((x - x_s)**2 + (y - y_s)**2 + (z - z_s)**2) / (2 * s_d**2)

# Vetor de variáveis
theta = [x, y, z]

# Hessiana
H = N * sp.Matrix([[sp.diff(sp.diff(l, i), j) for j in theta] for i in theta])

# Impressão simbólica
sp.pprint(H)
# Impressão simbólica
sp.pprint(H.inv())