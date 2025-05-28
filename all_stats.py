import time
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import psutil
from memory_profiler import memory_usage
from ucimlrepo import fetch_ucirepo
from stochastic_gradient_descent import StochGradientDecent


def measure_resources(func, *args, **kwargs):
    """
    Измеряет время выполнения, использование CPU и памяти для функции
    Возвращает: (время, максимальная память, средняя загрузка CPU)
    """

    mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=1)
    max_mem = max(mem_usage) - min(mem_usage)  # Разница по памяти в MiB

    start_time = time.perf_counter()
    p = psutil.Process()
    cpu_percent_start = p.cpu_percent(interval=None)
    result = func(*args, **kwargs)
    cpu_usage = p.cpu_percent(interval=None) - cpu_percent_start
    elapsed = time.perf_counter() - start_time
    
    return elapsed, max_mem, cpu_usage

def plot_weights_history(optimizer, title=""):
    """
    Строит график изменения весов
    """
    weights_history = np.array(optimizer.path)
    plt.figure(figsize=(10, 6))
    
    for i in range(weights_history.shape[1]):
        plt.plot(weights_history[:, i], label=f'Вес {i}')
    
    plt.title(f"{title}\nИзменение весов в процессе обучения")
    plt.xlabel("Итерация")
    plt.ylabel("Значение веса")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_optimization_history(optimizer, title: str):
    """
    Строит график изменения ошибки в процессе оптимизации
    :param optimizer: объект оптимизатора (SGD или GD)
    :param title: заголовок графика
    """
    if not hasattr(optimizer, 'path'):
        print("Ошибка: оптимизатор не сохраняет историю параметров")
        return
    
    # Вычисляем ошибку для каждой точки в path
    errors = []
    for weights in optimizer.path:
        if isinstance(optimizer, StochGradientDecent):
            error = optimizer._mse_loss(weights)
        else:
            error = optimizer.f(weights)
        errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(errors, label='Ошибка (MSE)')
    plt.xlabel('Итерация')
    plt.ylabel('Значение ошибки')
    plt.title(f'{title}\nФинальная ошибка: {errors[-1]:.4f}')
    plt.grid(color="Lightpink", visible=True)
    plt.legend()
    plt.show()

