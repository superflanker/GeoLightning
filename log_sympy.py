from sympy import symbols, pi, sqrt, exp, log, simplify, latex

# Define os símbolos
delta_dij, sigma_d = symbols('Delta_dij sigma_d', positive=True)

# Define a função de verossimilhança
L = (1 / sqrt(2 * pi * sigma_d**2)) * exp(- (delta_dij**2) / (2 * sigma_d**2))

# Calcula o logaritmo natural da verossimilhança
log_L = log(L)

# Simplifica a expressão resultante
log_L_simplificado = simplify(log_L)

# diff de diff


# Exibe a saída simbólica
print("Logaritmo da função de verossimilhança:")
print(latex(log_L_simplificado))
