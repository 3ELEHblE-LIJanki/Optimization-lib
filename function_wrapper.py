import hashlib
from typing import Callable, List

import dill


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
        self._cache = {}

    def _make_cache_key(self, args) -> str:
        """
        Создаёт хеш-ключ для кэша на основе аргументов.
        """
        try:
            raw = dill.dumps(args)
        except Exception:
            raw = repr(args).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    def __call__(self, *args) -> float:
        key = self._make_cache_key(args)
        if key in self._cache:
            return self._cache[key]

        self.count += 1
        result = self.f(*args)
        self._cache[key] = result
        return result

    
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
        self._cache.clear()

    def add_count(self, k):
        self.count += k