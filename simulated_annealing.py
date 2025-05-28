import numpy as np
import math

#alfa лучше чтобы было от 0.8 до 0.97 примерно
#тут почти всё сгенерировала нейросеть, потому что я уже слишком хотела спать (
def sa(f, bounds, T0=100, alpha=0.95, max_iter=1000):
    current = np.random.uniform(bounds[:,0], bounds[:,1])
    best = current.copy()
    for i in range(max_iter):
        T = T0 * (alpha**i)
        neighbor = current + np.random.normal(0, 1, len(bounds))
        neighbor = np.clip(neighbor, bounds[:,0], bounds[:,1])
        delta = f(neighbor) - f(current)
        if delta < 0 or np.random.rand() < math.exp(-delta/T):
            current = neighbor
            if f(current) < f(best):
                best = current.copy()
    return best