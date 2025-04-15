from typing import Callable, List
from functools import lru_cache

class FunctionWrapper:
    """
        Класс - обёртка над функцией, для подсчёта кол-ва вызовов
    """
    def __init__(self, f: Callable[[List[float]], float]):
        """
            :param f: - функция, которую мы хотим обернуть
        """
        self.f = f
        self.count = 0

    # @lru_cache(maxsize=None)
    def __call__(self, *args) -> float:
        """
            :param x: - точка, в которой мы хотим посчитать значение функции
        """
        self.count += 1
        res = self.f(*args)
        return res

    def call_without_memorization(self, *args) -> float:
        """
            Вызов функции без мемоизации (для графиков)
            :param x: - точка, в которой мы хотим посчитать значение функции
        """
        res = self.f(*args)
        return res
    
    def get_count(self) -> int:
        """
            :return: - кол-во вызовов функции
        """
        return self.count
    
    def clear(self):
        """
            сбросить кол-во вызовов функции
        """
        self.count = 0