from typing import Callable, List
from functools import lru_cache
import cloudpickle as pickle

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

    def __call__(self, *args) -> float:
        """
            :param x: - точка, в которой мы хотим посчитать значение функции
        """
        return self.f_cached(self.__serialize_data(*args))

    @staticmethod
    def __serialize_data(*args) -> bytes:
        return pickle.dumps(args)

    @lru_cache(maxsize=None)
    def f_cached(self, serialized: bytes) -> float:
        args = pickle.loads(serialized)
        self.count += 1
        return self.f(*args)

    
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