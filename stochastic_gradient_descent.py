from typing import Callable, List
from scipy.optimize import minimize

import numpy as np

from lrs import LRS, gradient, boundize, REG, regularity, regularity_L1, regularity_L2
from function_wrapper import FunctionWrapper


class StochGradientDecent:
    """
        Класс реализующий поиск максимума и минимума на основе градиентного спуска и переданных параметров
    """

    ACCEPTABLE_ACCURACY: float = 0.00001

    def __init__(self, learning_rate_scheduling: LRS, bounds: List[List[float]],
                 X_data: np.array, Y_data: np.array, batch_size: int = 1,
                 eps: float = ACCEPTABLE_ACCURACY, regular: REG = regularity_L1(0)):
        """
            :param learning_rate_scheduling: - выбранная модель поиска шага
            :param f: - функция для вычисления
            :param bounds: - границы функции
            :param eps: - точность подсчёта градиента
        """
        self.learning_rate_scheduling = learning_rate_scheduling
        self.bounds = bounds
        self.eps = eps
        self.batch_size = batch_size
        self.f = FunctionWrapper(self._mse_loss)
        self.regular = regular
        self.loss_history = []

        X_data_with_1 = np.c_[np.ones(X_data.shape[0]), X_data] # добавили в каждый вектор коэффициентов - 1 (типа константа)

        self.X_data = X_data_with_1
        self.Y_data = Y_data

    def __init(self, start: List[float]):
        self.x = start.copy()  # это веса для линейной регрессии
        self.path = []
        self.f.clear()

    def __find(self, start: np.array, max_iterations, op, gamma=None):
        self.__init(start)
        self.loss_history = []
        E = np.zeros(len(self.x))
        for i in range(max_iterations):
            self.path.append(self.x)
            # self.path.append(self.x)

            # градиент посчитали
            grad = np.array(self.calc_grad())

            # Функция потерь для данного batch
            h = self.learning_rate_scheduling(self.x, i, self.f, self.bounds)

            # После вычисления новых весов (для статистики функции потерь)
            current_loss = self._mse_loss(self.x)
            self.loss_history.append(current_loss)

            # Регулярность reg - вектор. Мы сразу считаем производную и добавим ее к градиенту
            reg = self.regular(self.x)

            # Вычисляем шаг
            if gamma == None:
                delta = h * (grad + reg)
            else:
                E = gamma * E + (1 - gamma) * np.square(grad)
                delta = np.divide(h * (grad + reg), np.sqrt(E))

            # делаем шаг спуска
            # # обрезаем по границам, если вылезло
            xx = boundize(op(self.x, delta), self.bounds)
            
            if np.linalg.norm(np.array(self.x) - np.array(xx)) < self.eps:
                break
            self.x = np.array(xx)

        return self.f(self.x)

    def calc_grad(self):
        # выбираем рандомные индексы -> batch
        batch_indices = np.random.choice(len(self.X_data), self.batch_size, replace=False)
        X_batch, y_batch = self.X_data[batch_indices], self.Y_data[batch_indices]

        # # считаем градиент по формуле (хардкод)
        # error = y_batch - X_batch.dot(self.x)
        # grad = -1 * 2 * X_batch.T.dot(error) / self.batch_size

        # считаем правильное количество вызовов функции в зависимости от batch
        mse_loss_batch = FunctionWrapper(lambda weights: np.mean((y_batch - X_batch.dot(weights)) ** 2))
        grad = gradient(mse_loss_batch, self.x, self.eps)
        self.f.add_count(mse_loss_batch.get_count())

        return grad


    def _mse_loss(self, weights) -> float:
        """MSE для датасета """
        X, y = self.X_data, self.Y_data
        return np.mean((y - X.dot(weights)) ** 2)

    def find_min(self, start: List[float], max_iterations: int, gamma: float = None) -> float:
        """
            :param  start: List[float] - стартовая точка, в которой начнём поиск
            :param  max_iterations: int - максимальное количество итераций спуска
            :return: - минимум полученный в ходе спуска
        """

        return self.__find(np.array(start), max_iterations, lambda x, y: x - y, gamma)

    def find_max(self, start: List[float], max_iterations: int, gamma: float = None) -> float:
        """
            :param start: - стартовая точка, в которой начнём поиск
            :param max_iterations: - максимальное количество итераций спуска
            :return: - максимум полученный в ходе спуска
        """
        return self.__find(start, max_iterations, lambda x, y: x + y, gamma)

    def get_bounds(self):
        return self.bounds

    def get_f(self):
        return self.f

    def get_path(self):
        return self.path

    def current_point(self):
        return [self.x, self.f(self.x)]

    def get_x(self):
        return self.x