import math
import random
from typing import Callable, List
from scipy.optimize import minimize

import numpy as np

from lrs import LRS, gradient
from function_wrapper import FunctionWrapper

class AnnealingDecent:
    """
        Класс реализующий поиск максимума и минимума на основе метода симуляции отжига и переданных параметров
    """

    ACCEPTABLE_ACCURACY: float = 0.00001


    def __init__(self,  f: FunctionWrapper, bounds, alpha: float, T0: float):
        """
            :param f: - функция для вычисления
            :param bounds: - границы функции
            :param alpha: - альфа
            :param T0: - T0
        """

        self.f = f
        self.bounds = bounds
        self.alpha = alpha
        self.T0 = T0

    def __init(self):
        self.path = []
        self.x = None

    def find_min(self, max_iter: int) -> List[float]:
        self.__init()
        current = [random.uniform(b[0], b[1]) for b in self.bounds]
        best = current.copy()
        
        for i in range(max_iter):
            # Сохраняем текущую точку в путь
            self.path.append(current.copy())
            self.x = current.copy()
            
            T = self.T0 * (self.alpha ** i)
            neighbor = [current[d] + random.gauss(0, 1) for d in range(len(self.bounds))] # генерируем соседа нашей точки с нормальным распределением
            neighbor = [max(min(neighbor[d], self.bounds[d][1]), self.bounds[d][0]) for d in range(len(self.bounds))] #что-то вроде boundize
            
            delta = self.f(neighbor) - self.f(current)
            T = max(T, 1e-10)
            exponent = -delta / T
            exponent = max(min(exponent, 700), -700)
            probability = math.exp(exponent)
            
            if delta < 0 or random.random() < probability: # условия перехода в новое состояние (зависят от температуры + текущего шага)
                current = neighbor.copy()
                if self.f(current) < self.f(best):
                    best = current.copy()
        
        return best


    def get_bounds(self):
        return self.bounds

    def get_f(self):
        return self.f

    def get_path(self):
        return self.path

    def current_point(self):
        return [self.x, self.f(self.x)]

