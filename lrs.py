from typing import Callable, List
import math

import numpy as np

from linear_desсent import LinearDecent
from function_wrapper import FunctionWrapper

EPS = 1e-8

def diff(f, x, eps, index: int):
    x_i = x.copy()
    x_i[index] += eps
    x_j = x.copy()
    x_j[index] -= eps
    return (f(x_i) - f(x_j)) / (2 * eps)

def __gradient(f, x, eps):
    grad = []
    for i in range(len(x)):
        grad.append(diff(f, x, eps, i))
    return grad

gradient = FunctionWrapper(__gradient)

def double_diff(f, x, eps, ind1, ind2):
    return diff(lambda x: diff(f, x, eps, ind1), x, eps, ind2)

def __hessian(f, x, eps):
    hess = []
    for i in range(len(x)):
        tmp = []
        for j in range(len(x)):
            tmp.append(double_diff(f, x, eps, i, j))
        hess.append(tmp)
    return hess

hessian = FunctionWrapper(__hessian)

def mult(x_1, x_2):
    res = 0
    for i in range(len(x_1)):
        res += x_1[i] * x_2[i]
    return res

def addv(x_1, x_2):
    return [x_1[i] + x_2[i] for i in range(len(x_1))]

def boundize(x, bounds):
    return [min(bounds[i][1], max(bounds[i][0], x[i])) for i in range(len(x))]


LRS = Callable[[tuple, int, FunctionWrapper, List[List[float]]], float]
'''
    Тип для Learning rate scheduling
'''

def armiho(c1: float, q: float) -> LRS:
    """
    Правило Армихо

    :param c1: - гиперпараметр

    :return: - LRS (learning rate scheduling) по правилу Армихо с заданными гипер-параметрами
    """
    return lambda x, _, f, f_bounds: _arm(c1, q, x, f, f_bounds)

def _arm(c1, q, x, f, f_bounds):
    ff = gradient(f, x, EPS)
    fff = [-ff_i for ff_i in ff]
    l = lambda a: f(x) + c1 * a * mult(fff, ff)
    a = 500000
    while l(a) < f(boundize(addv([a * ff_i for ff_i in fff], x), f_bounds)):
        a = q * a
    return a

def wolfe(c1: float = 1e-4, c2: float = 0.9) -> LRS:
    """
    Правило Вольфе

    :param c1: - гиперпараметр
    :param c2: - гиперпараметр

    :return: - LRS (learning rate scheduling) по правилу Вольфе с заданными гипер-параметрами
    """
    return lambda x, _, f, f_bounds: _wolfe(c1, c2, x, f, f_bounds)


def _wolfe(c1, c2, x, f, f_bounds):
    max_iterations = 100
    a = 1

    fx = f(x)
    grad = lambda x: np.array(gradient(f, x, EPS))
    grad_x_np = grad(x)

    # (1-Armijo) f(x + a * p) <= f(x) + c1 * a * graf(f)^T * p
    # (2-Curvature) grad(f(x + ap))^T * p >= c2 * grad(f(x))^T * p

    p = -grad_x_np
    right = grad_x_np.dot(p)  # grad(f(x))^T p - где p - прошлое направление - в моём случае -grad (??)

    for _ in range(max_iterations):
        x_new = boundize(np.array(x) + a * p, f_bounds)
        fx_new = f(x_new)
        grad_new = grad(x_new)

        # Проверка 1
        if fx_new > fx + c1 * a * right:
            a *= 0.5
            continue

        # Проверка 2
        if grad_new.dot(p) < c2 * right:
            a *= 1.5
            continue
        return a
    return a

def goldstein(c1: float = 0.1) -> LRS:
    """
    Правило Голдстейна

    :param c1: - гиперпараметр

    :return: - LRS (learning rate scheduling) по правилу Голдстейна с заданными гипер-параметрами
    """
    return lambda x, _, f, f_bounds: _goldstein(c1, x, f, f_bounds)


def _goldstein(c1, x, f, f_bounds):
    max_iterations = 100
    a = 1

    fx = f(x)
    grad = lambda x: np.array(gradient(f, x, EPS))
    grad_x_np = grad(x)

    # (1) f(x + a * p) <= f(x) + c1 * a * graf(f)^T * p
    # (2) f(x + a * p) >= f(x) + (1 - c1) * a * graf(f)^T * p

    p = -grad_x_np
    right = grad_x_np.dot(p)  # grad(f(x))^T p - где p grad (??)

    for _ in range(max_iterations):
        x_new = boundize(np.array(x) + a * p, f_bounds)
        fx_new = f(x_new)

        # Проверка 1
        if fx_new > fx + c1 * a * right:
            a *= 0.5
            continue

        grad_new = grad(x_new)
        # Проверка 2
        if fx_new < (1 - c1) * right:
            a *= 1.5
            continue
        return a
    return a


def constant(h0: float) -> LRS:
    """
    Постоянный метод планирования шага

    :param h0: - шаг

    :return: - Постоянный LRS (learning rate scheduling) с заданными гипер-параметрами
    """
    return lambda _x, _k, _f, _b: h0

def exponential_decay(h0: float, l: float) -> LRS:
    """
    Функциональный метод планирования шага (Экспоненциальное затухание)

    :h0: - начальный шаг
    :param l: - степень затухания

    :return: - Функциональный LRS (learning rate scheduling) с заданными гипер-параметрами
    """
    return lambda _x, k, _f, _b: h0 * math.e**(-l * k)

def polynomial_decay(a: float, b: float) -> LRS:
    """
    Функциональный метод планирования шага (Полиномиальное затухание)

    :param a: - гиперпараметр
    :param b: - гиперпараметр

    :return: - Функциональный LRS (learning rate scheduling) с заданными гипер-параметрами
    """
    return lambda _x, k, _f, _b: (1.0 / math.sqrt(k + 1)) * (b * k + 1)**(-a)


def linear_search(eps: float, max_steps_count: int, lin_algo) -> Callable:
    """
    Метод линейного поиска.
    :param eps: Точность поиска
    :param max_steps_count: максимальное число шагов
    :param lin_algo: алгоритм одномерного спуска
    :return: функция, выполняющая линейный поиск по направлению антиградиента
    """
    return lambda x, _, f, f_bounds: __linear_search(x, f, eps, max_steps_count, f_bounds, lin_algo)


def __linear_search(x: np.ndarray, f: Callable, eps: float, max_steps_count: int, f_bounds: List[List[float]], lin_algo):
    """
    Выполняет линейный поиск в направлении антиградиента.

    :param x: Текущая точка
    :param f: целевая функция
    :param eps: точность поиска
    :param max_steps_count: максимальное число шагов
    :param f_bounds: границы переменных
    :return: найденное значение шага
    """
    bounds = np.array([[0, np.inf]])
    grad = np.array(gradient(f, x, EPS))

    with np.errstate(divide='ignore', invalid='ignore'):
        candidates = np.array([
            (x - np.array(f_bounds)[:, 0]) / grad,
            (x - np.array(f_bounds)[:, 1]) / grad
        ])

    min_positive_candidate = np.min(candidates[candidates > 0], initial=np.inf)
    bounds[0, 1] = min(bounds[0, 1], min_positive_candidate)

    objective_function = lambda h: f(x - h * grad)

    linear_searcher = LinearDecent(objective_function, bounds, eps, lin_algo)
    linear_searcher.find_min(bounds[0, 0], max_steps_count)

    return linear_searcher.get_path()[-1][0]
