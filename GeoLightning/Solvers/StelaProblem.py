"""
    EELT 7019 - Inteligência Artificial Aplicada
    Classe Problema do Algoritmo Stela
    Autor: Augusto Mathias Adams <augusto.adams@ufpr.br>

    Este problema modela a estimação espaço-temporal de eventos a partir de
    detecções em sensores distribuídos, como no caso de descargas atmosféricas.
    Cada vetor de solução representa um conjunto de coordenadas (lat, lon, alt)
    de eventos candidatos, cuja qualidade é avaliada via a função de verossimilhança
    espacial e temporal calculada pelo algoritmo STELA.
"""
import numpy as np
from mealpy import FloatVar, Problem
from GeoLightning.Stela.Stela import stela
from GeoLightning.Utils.Constants import SIGMA_D, \
    EPSILON_D, \
    EPSILON_T, \
    LIMIT_D, \
    CLUSTER_MIN_PTS, \
    MAX_DISTANCE


class StelaProblem(Problem):
    def __init__(self,
                 bounds,
                 minmax,
                 pontos_de_chegada: np.ndarray,
                 tempos_de_chegada: np.ndarray,
                 sistema_cartesiano: bool = False,
                 sigma_d: np.float64 = SIGMA_D,
                 epsilon_t: np.float64 = EPSILON_T,
                 epsilon_d: np.float64 = EPSILON_D,
                 limit_d: np.float64 = LIMIT_D,
                 max_d: np.float64 = MAX_DISTANCE,
                 min_pts: np.int32 = CLUSTER_MIN_PTS,
                 **kwargs):
        """
            Inicializa uma instância do problema STELA para uso com algoritmos de otimização
            meta-heurística da biblioteca MEALPY.

            Este problema modela a estimação espaço-temporal de eventos a partir de
            detecções em sensores distribuídos, como no caso de descargas atmosféricas.
            Cada vetor de solução representa um conjunto de coordenadas (lat, lon, alt)
            de eventos candidatos, cuja qualidade é avaliada via a função de verossimilhança
            espacial e temporal calculada pelo algoritmo STELA.

            Args:
                bounds (list): Lista contendo dois arrays NumPy [lower_bounds, upper_bounds]
                            de forma (3M,), onde M é o número de eventos candidatos.
                            Cada tripla (lat, lon, alt) representa um evento.
                minmax (str): String indicando o tipo de otimização ('min' ou 'max').
                pontos_de_chegada (np.ndarray): Matriz de forma (N, 3) com as coordenadas
                                                geográficas dos N sensores (latitude, longitude, altitude).
                tempos_de_chegada (np.ndarray): Vetor de forma (N,) com os tempos de chegada
                                                dos sinais em cada sensor.
                sistema_cartesiano (bool): Indica se o sistema de coordenadas usado é cartesiano.
                                        Caso False, assume coordenadas geodésicas.
                sigma_d (np.float64): Desvio padrão do erro espacial nas medidas de distância (TOA).
                epsilon_t (np.float64): Tolerância máxima para agrupamento temporal.
                epsilon_d (np.float64): Tolerância máxima para agrupamento espacial.
                limit_d (np.float64): Raio de busca em torno de um evento durante o refinamento.
                max_d (np.float64): Distância máxima admissível entre eventos e detecções.
                min_pts (np.int32): Número mínimo de detecções para que um cluster seja válido.
                **kwargs: Parâmetros adicionais aceitos pela superclasse `Problem`.

            Atributos:
                clusters_espaciais (np.ndarray): Vetor de rótulos de clusterização espacial.
                centroides (np.ndarray): Coordenadas dos centróides espaciais obtidos.
                detectores (np.ndarray): Máscara indicando os detectores associados.
        """
        # parâmetros passados
        self.pontos_de_chegada = pontos_de_chegada
        self.tempos_de_chegada = tempos_de_chegada
        self.sistema_cartesiano = sistema_cartesiano
        self.sigma_d = sigma_d
        self.epsilon_d = epsilon_d
        self.epsilon_t = epsilon_t
        self.limit_d = limit_d
        self.max_d = max_d
        self.min_pts = min_pts

        # variáveis internas de controle
        self.fitness_values = list()
        self.stela_ub = list()
        self.stela_lb = list()
        self.stela_centroides = list()
        self.stela_clusters_espaciais = list()
        self.stela_novas_solucoes = list()
        self.stela_detectores = list()
        # variáveis de resposta
        self.clusters_espaciais = - \
            np.ones(pontos_de_chegada.shape[0], dtype=np.int32)
        self.centroides = -np.ones(pontos_de_chegada.shape)
        self.detectores = -np.ones(pontos_de_chegada.shape[0], dtype=np.int32)
        self.n_dims = pontos_de_chegada.shape[1]
        super().__init__(bounds, minmax, solution_encoding="float", **kwargs)

    def restart_search_space(self):
        """
            Atualiza os limites inferior (`lb`) e superior (`ub`) do espaço de busca com base 
            na melhor solução encontrada até o momento pela função de avaliação do problema.

            Esta função é utilizada para refinar iterativamente a busca em torno das regiões 
            mais promissoras do espaço de soluções. A estratégia consiste em identificar a 
            melhor verossimilhança registrada (máxima ou mínima, conforme `minmax`), 
            e reconfigurar os limites do problema e os dados internos do STELA 
            de acordo com essa configuração mais favorável.

            Após a atualização, os vetores auxiliares de controle são esvaziados, preparando
            a próxima fase da otimização.

            Efeitos colaterais:
                - Atualiza `self.lb` e `self.ub`.
                - Atualiza `self.centroides`, `self.clusters_espaciais` e `self.detectores`.
                - Reinicializa as listas auxiliares de controle.
        """
        if len(self.fitness_values) > 0:
            fitness_values = np.array(self.fitness_values)
            # encontrando a melhor solução dentre as sugeridas
            if self.minmax == "min":
                best_fitness_index = np.argwhere(
                    fitness_values == np.min(fitness_values)).flatten()[0]
            else:
                best_fitness_index = np.argwhere(
                    fitness_values == np.max(-fitness_values)).flatten()[0]
            # ajustando os limites 
            bounds = FloatVar(ub=self.stela_ub[best_fitness_index],
                              lb=self.stela_lb[best_fitness_index])
            self.set_bounds(bounds)
            # guardando informações finais
            self.clusters_espaciais = self.stela_clusters_espaciais[best_fitness_index]
            self.centroides = self.stela_centroides[best_fitness_index]
            self.detectores = self.stela_detectores[best_fitness_index]
            # reiniciando as listas
            self.fitness_values = list()
            self.stela_ub = list()
            self.stela_lb = list()
            self.stela_centroides = list()
            self.stela_clusters_espaciais = list()
            self.stela_novas_solucoes = list()
            self.stela_detectores = list()

    def evaluate(self, solution):
        return [self.obj_func(solution)]

    def get_best_solution(self):
        """
        Retorna a melhor solução
        Args:
            None
        Returns:
            Tuple =>
                nova_solucao (np.ndarray): a melhor solução calculada
                fitness_value (np.float64): o melhr fitness
        """
        if not self.fitness_values:
            return None, None
        idx = np.argmin(self.fitness_values) if self.minmax == "min" else np.argmax(
            self.fitness_values)

        return self.stela_novas_solucoes[idx], self.fitness_values[idx]

    def obj_func(self, solution):
        """
            Função objetivo para o problema STELA.

            Avalia a qualidade de uma solução candidata com base na
            verossimilhança espaço-temporal dos eventos estimados. Essa verossimilhança
            é calculada por meio do algoritmo STELA, que considera tanto os tempos de chegada
            quanto as posições dos detectores para agrupar e refinar eventos.

            Args:
                solution (np.ndarray): vetor unidimensional representando uma sequência 
                                    de coordenadas (lat, lon, alt) empilhadas.

            Returns:
                list: valor escalar da função objetivo (fitness), negativo se for 
                problema de maximização.
        """
        # Converte o vetor linear para o formato (M, 3)
        solucoes = self.decode_solution(solution)

        # Executa o algoritmo STELA
        (lb,
         ub,
         centroides,
         detectores,
         clusters_espaciais,
         novas_solucoes,
         verossimilhanca) = stela(solucoes,
                                  self.tempos_de_chegada,
                                  self.pontos_de_chegada,
                                  self.clusters_espaciais,
                                  self.sistema_cartesiano,
                                  self.sigma_d,
                                  self.epsilon_t,
                                  self.epsilon_d,
                                  self.limit_d,
                                  self.max_d,
                                  self.min_pts)

        # Armazena resultados auxiliares para possível refinamento posterior
        self.fitness_values.append(-verossimilhanca if self.minmax ==
                                   "max" else verossimilhanca)
        self.stela_ub.append(ub)
        self.stela_lb.append(lb)
        self.stela_centroides.append(centroides)
        self.stela_clusters_espaciais.append(clusters_espaciais)
        self.stela_novas_solucoes.append(novas_solucoes)
        self.stela_detectores.append(detectores)

        # Retorna a verossimilhança como valor de fitness (negativa para problema
        # de maximização)
        return -verossimilhanca if self.minmax == "max" else verossimilhanca
