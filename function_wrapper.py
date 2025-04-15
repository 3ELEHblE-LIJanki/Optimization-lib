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
        res = self.f(*args)
        if ("memo" not in self.__dict__):
            # self.__dict__["memo"] = {}
            return res
        if (args in self.memo):
            return self.memo[args]

        self.memo[args] = res
        return res
    
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