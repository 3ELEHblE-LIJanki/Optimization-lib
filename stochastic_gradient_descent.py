from typing import Callable, List
from scipy.optimize import minimize

import numpy as np
from winerror import DXGI_ERROR_MORE_DATA

from lrs import LRS, gradient_stochastic, gradient
from function_wrapper import FunctionWrapper


class StochGradientDecent:
    """
        Класс реализующий поиск максимума и минимума на основе градиентного спуска и переданных параметров
    """

    ACCEPTABLE_ACCURACY: float = 0.00001

    def __init__(self, learning_rate_scheduling: LRS, bounds: List[List[float]],
                 X_data: np.array, Y_data: np.array, batch_size: int = 1,
                 eps: float = ACCEPTABLE_ACCURACY):
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

        X_data_with_1 = np.c_[np.ones(X_data.shape[0]), X_data]

        self.X_data = X_data_with_1
        self.Y_data = Y_data

    def __init(self, start: List[float]):
        self.x = start.copy()  # это веса для линейной регрессии
        self.path = []

    def __find(self, start: np.array, max_iterations, op):
        self.__init(start)
        for i in range(max_iterations):
            self.path.append(self.x)

            # выбираем рандомные индексы -> batch
            # rand = np.random.RandomState(i)

            batch_indices = np.random.choice(len(self.X_data), self.batch_size, replace=False)
            X_batch, y_batch = self.X_data[batch_indices], self.Y_data[batch_indices]

            # считаем градиент по формуле
            error = y_batch - X_batch.dot(self.x)  # добавить (1, x1, x2...) !!!!!!
            grad = 2 * X_batch.T.dot(error) / self.batch_size
            # grad = gradient_stochastic(X_batch, y_batch, self.x)

            mse_loss_batch = lambda weights: np.mean((y_batch - X_batch.dot(weights)) ** 2)
            # grad = gradient(mse_loss_batch, self.x, self.eps)

            # Функция потерь для данного batch
            # print(grad)
            h = self.learning_rate_scheduling(self.x, i, mse_loss_batch, self.bounds)

            # делаем шаг спуска
            # new_x = self.x - h * grad
            # # обрезаем по границам, если вылезло
            # new_x = np.clip(new_x, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            xx = []
            for j in range(len(self.x)):
                coord = op(self.x[j], h * grad[j])
                print(coord)
                coord = max(coord, self.bounds[j][0])
                coord = min(coord, self.bounds[j][1])
                xx.append(coord)

            # ?????????
            if np.linalg.norm(np.array(self.x) - np.array(xx)) < self.eps:
                break
            self.x = np.array(xx)
            print(self.x)

            # print("current: " + str(i) + " " + str(self.f(self.x)))
        return self.f(self.x)

    def _mse_loss(self, weights) -> float:
        """MSE для датасета """
        X, y = self.X_data, self.Y_data
        return np.mean((y - X.dot(weights)) ** 2)

    def find_min(self, start: List[float], max_iterations: int) -> float:
        """
            :param  start: List[float] - стартовая точка, в которой начнём поиск
            :param  max_iterations: int - максимальное количество итераций спуска
            :return: - минимум полученный в ходе спуска
        """

        return self.__find(np.array(start), max_iterations, lambda x, y: x - y)

    def find_max(self, start: List[float], max_iterations: int) -> float:
        """
            :param start: - стартовая точка, в которой начнём поиск
            :param max_iterations: - максимальное количество итераций спуска
            :return: - максимум полученный в ходе спуска
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
