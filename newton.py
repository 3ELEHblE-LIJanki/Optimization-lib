from typing import List
from function_wrapper import FunctionWrapper
from lrs import LRS, hessian, gradient, boundize
import numpy as np


class NewtonCG:
    """
        Класс реализующий поиск максимума и минимума на основе градиентного спуска и переданных параметров
    """

    ACCEPTABLE_ACCURACY: float = 0.00001

    def __init__(self, learning_rate_scheduling: LRS, f: FunctionWrapper, bounds: List[List[float]],
                 eps: float = ACCEPTABLE_ACCURACY):
        """
            :param learning_rate_scheduling: - выбранная модель поиска шага
            :param f: - функция для вычисления
            :param bounds: - границы функции
            :param eps: - точность подсчёта градиента
        """
        self.learning_rate_scheduling = learning_rate_scheduling
        self.proto_f = f
        # self.f = lambda x: np.linalg.norm(np.array(gradient(f, x, eps)))
        self.f = f
        self.bounds = bounds
        self.eps = eps

    def __init(self, start: List[float]):
        self.x = start.copy()
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
            H = np.array(hessian(self.f, self.x, self.eps))
            l = self.learning_rate_scheduling(self.x, i, self.f, self.bounds)
            self.x = boundize(np.array(self.x) - l * np.linalg.inv(H).dot(G), self.bounds)

        return self.f(self.x)

    def get_bounds(self):
        return self.bounds

    def get_f(self):
        return self.proto_f

    def get_path(self):
        return self.path

    def current_point(self):
        return [self.x, self.f(self.x)]
