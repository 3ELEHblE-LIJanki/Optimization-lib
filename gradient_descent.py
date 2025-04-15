from typing import Callable, List
from scipy.optimize import minimize

import numpy as np

from lrs import LRS, gradient
from function_wrapper import FunctionWrapper

class GradientDecent:
    """
        Класс реализующий поиск максимума и минимума на основе градиентного спуска и переданных параметров
    """

    ACCEPTABLE_ACCURACY: float = 0.00001


    def __init__(self, learning_rate_scheduling: LRS, f: FunctionWrapper, bounds: List[List[float]],
                 eps: float = ACCEPTABLE_ACCURACY):
        """
            learning_rate_scheduling - выбранная модель поиска шага
            max_iterations - максимальное число итераций (чтобы не зациклиться)
            eps: - точность подсчёта градиента
        """
        self.learning_rate_scheduling = learning_rate_scheduling
        self.f = f
        self.bounds = bounds
        self.eps = eps

    def __init(self, start: List[float]):
        self.x = start.copy()
        self.path = []

    def __find(self, start, max_iterations, op):
        self.__init(start)
        for i in range(max_iterations):
            h = self.learning_rate_scheduling(self.x, i, self.f, self.bounds)
            self.path.append(self.x)
            grad = gradient(self.f, self.x, self.eps)
            xx = []
            for j in range(len(self.x)):
                coord = op(self.x[j], h * grad[j])
                coord = max(coord, self.bounds[j][0])
                coord = min(coord, self.bounds[j][1])
                xx.append(coord)
            if np.linalg.norm(np.array(self.x) - np.array(xx)) < self.eps:
                break
            self.x = xx
        return self.f(self.x)

    def find_min(self, start: List[float], max_iterations: int) -> float:
        """
            start: List[float] - стартовая точка, в которой начнём поиск
            max_iterations: int - максимальное количество итераций спуска
            return - минимум полученный в ходе спуска
        """
        return self.__find(start, max_iterations, lambda x, y: x - y)
        

    def find_max(self, start: List[float], max_iterations: int) -> float:
        """
            start: List[float] - стартовая точка, в которой начнём поиск
            max_iterations: int - максимальное количество итераций спуска
            return - максимум полученный в ходе спуска
        """
        return self.__find(start, max_iterations, lambda x, y: x + y)

    def get_bounds(self):
        return self.bounds

    def get_f(self):
        return self.f

    def get_path(self):
        return self.path

    def current_point(self):
        return [self.x, self.f(self.x)]


class SimpyWrapper:
    """
        Класс реализующий поиск максимума и минимума на основе scipy.optimize
    """

    ACCEPTABLE_ACCURACY: float = 0.00001


    def __init__(self, f: FunctionWrapper, bounds: List[List[float]],
                 eps: float = ACCEPTABLE_ACCURACY):
        """
            method: str - название метода из библиотеки scipy.optimize
            bounds: - границы исследования функции
            eps: - точность подсчёта градиента
        """
        self.f = f
        self.bounds = bounds
        self.eps = eps

    def find_min(self, method: str, start: List[float], max_iterations: int, gradient=None, hessian=None) -> float:
        """
            method: str - метод, который будет использован для поиска минимума
            start: List[float] - стартовая точка, в которой начнём поиск
            max_iterations: int - максимальное количество итераций спуска
            return - минимум полученный в ходе спуска
        """
        self.path=[start]
        self.res = minimize(
            x0=start,
            fun=self.f,
            method=method,
            callback=lambda x: self.path.append(x.tolist()),
            options={'maxiter': max_iterations},
            jac=gradient,
            hess=hessian
        )
        return self.f(self.res.x)

    def get_bounds(self):
        return self.bounds

    def get_f(self):
        return self.f

    def get_path(self):
        return self.path

    def current_point(self):
        return [self.res.x, self.f(self.res.x)]