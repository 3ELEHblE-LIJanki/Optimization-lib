from typing import List
from function_wrapper import FunctionWrapper
from lrs import LRS, _wolfe, gradient, boundize
import numpy as np


class BFGS:
    """
        Класс реализующий поиск максимума и минимума на основе градиентного спуска и переданных параметров
    """

    ACCEPTABLE_ACCURACY: float = 0.00001

    def __init__(self, f: FunctionWrapper, bounds: List[List[float]],
                 eps: float = ACCEPTABLE_ACCURACY):
        """
            :param f: - функция для вычисления
            :param bounds: - границы функции
            :param eps: - точность подсчёта градиента
        """
        self.learning_rate_scheduling = lambda x, p: _wolfe(0.0001, 0.4, x, f, bounds, p) 
        self.f = f
        self.bounds = bounds
        self.eps = eps
        self.I = np.eye(len(bounds))

    def __init(self, start: List[float]):
        self.x = start.copy()
        self.C = self.I.copy()
        self.path = []

    def find_min(self, start, max_iterations):
        """
            :param start: List[float] - стартовая точка, в которой начнём поиск
            :param max_iterations: int - максимальное количество итераций спуска
            :return: - минимум полученный в ходе спуска
        """
        self.__init(start)
        for i in range(max_iterations):
            self.path.append(self.x)
            G = np.array(gradient(self.f, self.x, self.eps))
            if np.linalg.norm(G) <= self.eps:
                break
            P = self.C.dot(-G)
            l = self.learning_rate_scheduling(self.x, P)
            new_x = boundize(self.x + l * P, self.bounds)
            s = np.array(new_x) - np.array(self.x)
            y = np.array(gradient(self.f, new_x, self.eps)) - G
            p = 1 / y.dot(s)
            new_C = (self.I - p * s.dot(y)).dot(self.C).dot(self.I - p * y.dot(s)) + p * s.dot(s)
            self.C = new_C
            self.x = new_x
            

        return self.f(self.x)

    def get_bounds(self):
        return self.bounds

    def get_f(self):
        return self.f

    def get_path(self):
        return self.path

    def current_point(self):
        return [self.x, self.f(self.x)]
