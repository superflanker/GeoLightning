"""
EELT 7019 - Applied Artificial Intelligence
Utils Tests
Author: Augusto Mathias Adams <augusto.adams@ufpr.br>
"""

import unittest
import numpy as np
from GeoLightning.Utils.Utils import (
    coordenadas_em_radianos,
    coordenadas_em_radianos_batelada,
    distancia_cartesiana_entre_pontos,
    distancia_esferica_entre_pontos,
    computa_distancia,
    computa_tempos_de_origem
)
from GeoLightning.Utils.Constants import AVG_EARTH_RADIUS, AVG_LIGHT_SPEED

class TestUtils(unittest.TestCase):
    """
    Classe de testes para as funções utilitárias em GeoLightning.Utils.
    """

    def setUp(self):
        """
        Configuração inicial para os testes. Define dados de exemplo.
        """
        # Pontos de exemplo para testes de distância e coordenadas
        self.ponto_cartesiano_1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.ponto_cartesiano_2 = np.array([3.0, 4.0, 0.0], dtype=np.float64) # Distância 5.0
        self.ponto_cartesiano_3 = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        # Coordenadas geográficas (graus e radianos) e altitudes
        # Exemplo: Reitoria UFPR -25.4284, -49.2733, 915m
        self.lat_lon_graus_1 = np.array([-25.4284, -49.2733, 915.0], dtype=np.float64)
        # Exemplo: Aeroporto Afonso Pena -25.5323, -49.1764, 907m
        self.lat_lon_graus_2 = np.array([-25.5323, -49.1764, 907.0], dtype=np.float64)
        
        # Coordenadas geográficas já em radianos para usar com distancia_esferica_entre_pontos
        self.lat_rad_1 = np.deg2rad(self.lat_lon_graus_1[0])
        self.lon_rad_1 = np.deg2rad(self.lat_lon_graus_1[1])
        self.alt_1 = self.lat_lon_graus_1[2]
        self.ponto_esferico_rad_1 = np.array([self.lat_rad_1, self.lon_rad_1, self.alt_1], dtype=np.float64)

        self.lat_rad_2 = np.deg2rad(self.lat_lon_graus_2[0])
        self.lon_rad_2 = np.deg2rad(self.lat_lon_graus_2[1])
        self.alt_2 = self.lat_lon_graus_2[2]
        self.ponto_esferico_rad_2 = np.array([self.lat_rad_2, self.lon_rad_2, self.alt_2], dtype=np.float64)

        # Dados para computa_tempos_de_origem
        self.solucoes_ex = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0],[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]], dtype=np.float64)
        self.clusters_espaciais_ex = np.array([0, 1, 0, 1], dtype=np.int32) # Detectores 0 e 2 associados à solução 0, 1 e 3 à solução 1
        self.tempos_de_chegada_ex = np.array([5.0, 5.0, 6.0, 6.0], dtype=np.float64)
        self.pontos_de_deteccao_ex = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [1.0, 0.0, 0.0], [11.0, 10.0, 10.0]], dtype=np.float64)
        
        # Dados para testes de arrays
        self.array_simples = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        self.array_float = np.array([1.1, 2.2, 3.3], dtype=np.float64)


    # Testes para coordenadas_em_radianos
    def test_coordenadas_em_radianos(self):
        """Testa a conversão de coordenadas individuais para radianos."""
        # A altitude não deve ser convertida
        coords_graus = np.array([90.0, 180.0, 1000.0], dtype=np.float64)
        expected_rads = np.array([np.pi / 2, np.pi, 1000.0], dtype=np.float64)
        converted_coords = coordenadas_em_radianos(coords_graus)
        np.testing.assert_almost_equal(converted_coords, expected_rads, decimal=6)

        # Teste com valores negativos
        coords_graus_neg = np.array([-45.0, -90.0, 500.0], dtype=np.float64)
        expected_rads_neg = np.array([-np.pi / 4, -np.pi / 2, 500.0], dtype=np.float64)
        converted_coords_neg = coordenadas_em_radianos(coords_graus_neg)
        np.testing.assert_almost_equal(converted_coords_neg, expected_rads_neg, decimal=6)

    def test_coordenadas_em_radianos_batelada(self):
        """Testa a conversão de um array de coordenadas para radianos (batelada)."""
        coords_graus_batch = np.array([
            [90.0, 180.0, 100.0],
            [45.0, 90.0, 200.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float64)
        expected_rads_batch = np.array([
            [np.pi / 2, np.pi, 100.0],
            [np.pi / 4, np.pi / 2, 200.0],
            [0.0, 0.0, 0.0]
        ], dtype=np.float64)
        converted_coords_batch = coordenadas_em_radianos_batelada(coords_graus_batch)
        np.testing.assert_almost_equal(converted_coords_batch, expected_rads_batch, decimal=6)

        # Teste com array vazio
        empty_array = np.array([], dtype=np.float64).reshape(0, 3)
        converted_empty = coordenadas_em_radianos_batelada(empty_array)
        self.assertEqual(converted_empty.shape, (0, 3))


    # Testes para distancia_cartesiana_entre_pontos
    def test_distancia_cartesiana_entre_pontos(self):
        """Testa o cálculo da distância euclidiana 3D."""
        dist = distancia_cartesiana_entre_pontos(self.ponto_cartesiano_1, self.ponto_cartesiano_2)
        self.assertAlmostEqual(dist, 5.0, places=9) # sqrt(3^2 + 4^2 + 0^2) = 5

        dist_self = distancia_cartesiana_entre_pontos(self.ponto_cartesiano_1, self.ponto_cartesiano_1)
        self.assertAlmostEqual(dist_self, 0.0, places=9)
        
        dist_3d = distancia_cartesiana_entre_pontos(self.ponto_cartesiano_1, self.ponto_cartesiano_3)
        self.assertAlmostEqual(dist_3d, np.sqrt(3.0), places=9)


    # Testes para distancia_esferica_entre_pontos
    def test_distancia_esferica_entre_pontos(self):
        dist_expected = 275.76 # 
        dist = distancia_esferica_entre_pontos(self.ponto_esferico_rad_1, self.ponto_esferico_rad_2)
        
        self.assertAlmostEqual(dist, 275.76, delta=1.0) # Delta de 100 metros para aceitar a aproximação
                                                          # e o valor da distância real.

        # Teste de distância zero (mesmo ponto)
        dist_zero = distancia_esferica_entre_pontos(self.ponto_esferico_rad_1, self.ponto_esferico_rad_1)
        self.assertAlmostEqual(dist_zero, 0.0, places=9)

        # Teste com diferença de altitude apenas (latitude e longitude iguais)
        ponto_a = np.array([0.0, 0.0, 0.0], dtype=np.float64) # Lat/Lon em radianos para simplificar
        ponto_b = np.array([0.0, 0.0, 100.0], dtype=np.float64)
        dist_alt_only = distancia_esferica_entre_pontos(ponto_a, ponto_b)
        self.assertAlmostEqual(dist_alt_only, 0.0, places=9) # Diferença de altitude de 100m


    # Testes para computa_distancia
    def test_computa_distancia_cartesiana(self):
        """Testa computa_distancia no modo cartesiano."""
        dist = computa_distancia(self.ponto_cartesiano_1, self.ponto_cartesiano_2, True)
        self.assertAlmostEqual(dist, 5.0, places=9)

    def test_computa_distancia_esferica(self):
        """Testa computa_distancia no modo esférico."""
        dist = computa_distancia(self.ponto_esferico_rad_1, self.ponto_esferico_rad_2, False)
        # Reutiliza a verificação do teste de distancia_esferica_entre_pontos
        self.assertAlmostEqual(dist, 275.76, delta=100) 


    # Testes para computa_tempos_de_origem
    def test_computa_tempos_de_origem_cartesiano(self):
        """Testa o cálculo dos tempos de origem em sistema cartesiano."""
        # Detector 0: associado a solução 0 (0,0,0). Ponto detecção (0,0,0). Dist = 0. Tempo_origem = 5 - 0/c = 5
        # Detector 1: associado a solução 1 (10,10,10). Ponto detecção (10,10,10). Dist = 0. Tempo_origem = 5 - 0/c = 5
        # Detector 2: associado a solução 0 (0,0,0). Ponto detecção (1,0,0). Dist = 1. Tempo_origem = 6 - 1/c
        # Detector 3: associado a solução 1 (10,10,10). Ponto detecção (11,10,10). Dist = 1. Tempo_origem = 6 - 1/c

        tempos_origem = computa_tempos_de_origem(self.solucoes_ex,
                                                 self.clusters_espaciais_ex,
                                                 self.tempos_de_chegada_ex,
                                                 self.pontos_de_deteccao_ex,
                                                 True) # Cartesiano

        expected_tempos_origem = np.array([
            self.tempos_de_chegada_ex[0] - 0.0 / AVG_LIGHT_SPEED, # Distância 0
            self.tempos_de_chegada_ex[1] - 0.0 / AVG_LIGHT_SPEED, # Distância 0
            self.tempos_de_chegada_ex[2] - 1.0 / AVG_LIGHT_SPEED, # Distância 1
            self.tempos_de_chegada_ex[3] - 1.0 / AVG_LIGHT_SPEED  # Distância 1
        ], dtype=np.float64)
        
        np.testing.assert_almost_equal(tempos_origem, expected_tempos_origem, decimal=9)

    def test_computa_tempos_de_origem_vazio(self):
        """Testa computa_tempos_de_origem com entradas vazias."""
        empty_solucoes = np.array([], dtype=np.float64).reshape(0, 3)
        empty_clusters = np.array([], dtype=np.int32)
        empty_tempos_chegada = np.array([], dtype=np.float64)
        empty_pontos_deteccao = np.array([], dtype=np.float64).reshape(0, 3)

        tempos_origem = computa_tempos_de_origem(empty_solucoes,
                                                 empty_clusters,
                                                 empty_tempos_chegada, 
                                                 empty_pontos_deteccao, 
                                                 True)
        self.assertEqual(len(tempos_origem), 0)
