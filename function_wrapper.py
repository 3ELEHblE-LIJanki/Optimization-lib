from typing import Callable, List

class FunctionWrapper:
    """
        Класс - обёртка над функцией, для подсчёта кол-ва вызовов
    """
    def __init__(self, f: Callable[[List[float]], float]):
        """
            :param f: - функция, которую мы хотим обернуть
        """
        self.memo = None
        self.f = f
        self.count = 0

    def __call__(self, *args) -> float:
        """
            :param x: - точка, в которой мы хотим посчитать значение функции
        """
        if self.memo is not None and args in self.memo:
            return self.memo[args]
        
        self.count += 1
        res = self.f(*args)
        if self.memo is not None:
            self.memo[args] = res
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