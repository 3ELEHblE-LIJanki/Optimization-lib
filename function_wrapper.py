from typing import Callable, List

class FunctionWrapper:
    """
        Класс - обёртка над функцией, для подсчёта кол-ва вызовов
    """
    def __init__(self, f: Callable[[List[float]], float]):
        """
            f: - функция, которую мы хотим обернуть
        """
        self.f = f
        self.count = 0

    def __call__(self, *args) -> float:
        """
            x: - точка, в которой мы хотим посчитать значение функции
        """
        self.count += 1
        return self.f(*args)
    
    def get_count(self) -> int:
        """
            return: - кол-во вызовов функции
        """
        return self.count
    
    def clear(self):
        """
            сбросить кол-во вызовов функции
        """
        self.count = 0