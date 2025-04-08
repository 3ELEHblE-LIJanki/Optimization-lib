from math import sqrt
from typing import Callable, List

from gradient_descent import FunctionWrapper
import numpy as np

# работает только для функций 1d
# Расчитываю что он будет рисовать границы

class LinearDecent:
    def __init__(self, f: FunctionWrapper, bounds: List[List[float]], eps, lin_algo):
        self.f = f
        self.eps = eps
        self.bounds = bounds
        self.path = []
        self.x = -100
        self.lin_algo = lin_algo

    def __init(self,  start: float):
        if isinstance(start, list):
            if len(start) > 1:
                raise TypeError("Only for R -> R functions")
            self.path.append(start)
            start = start[0]
        else:
            self.path.append([start])
        self.x : float = start

    def find_min(self, start: float | list[float], max_steps_count: int):
        self.__init(start)
        l = self.bounds[0][0]
        r = self.bounds[0][1]
        self.path.extend(self.lin_algo(self.f, l, r, self.eps, max_steps_count))
        return self.f(self.path[-1])

    def find_max(self, start: float | list[float], max_steps_count: int):
        self.__init(start)
        l = self.bounds[0][0]
        r = self.bounds[0][1]
        self.path.extend(self.lin_algo(lambda x: -1 * self.f(x), l, r, self.eps, max_steps_count))
        return self.f(self.path[-1])


    def get_bounds(self):
        return self.bounds

    def get_f(self):
        return self.f

    def get_path(self):
        return self.path