import math
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

    def find_min(self,  max_iter: int) -> float:
        """
            :param  max_iterations: int - максимальное количество итераций спуска
            :return: - минимум полученный в ходе спуска
        """

        self.__init()
        current = np.random.uniform(self.bounds[:,0], self.bounds[:,1])
        best = current.copy()
        for i in range(max_iter):
            if (self.path.__len__ != 0): 
                self.path.append(self.x)

            T = self.T0 * (self.alpha**i)
            neighbor = current + np.random.normal(0, 1, len(self.bounds))
            neighbor = np.clip(neighbor, self.bounds[:,0], self.bounds[:,1])
            delta = self.f(neighbor) - self.f(current)
            if delta < 0 or np.random.rand() < math.exp(-delta/T):
                current = neighbor
                self.x = current;
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

