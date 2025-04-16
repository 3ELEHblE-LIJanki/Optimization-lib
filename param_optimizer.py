import logging
from functools import partial
from typing import List, Type

import optuna

from function_wrapper import FunctionWrapper
from gradient_descent import GradientDecent
from linear_desсent import LinearDecent
from lrs import gradient, LRS
from newton import NewtonCG

ACCEPTABLE_ACCURACY: float = 0.00001
optuna.logging.set_verbosity(logging.CRITICAL)
class ParamOptimizer:
    def __init__(self, algo: Type[GradientDecent|LinearDecent|NewtonCG], f: FunctionWrapper, bounds: List[List[float]],
                 eps: float = ACCEPTABLE_ACCURACY):
        self._f = f
        self._algo_partial = partial(algo, f=self._f, bounds=bounds, eps=eps)

    def optimize(self, lrs: LRS, params: dict[str, tuple[float, float]], start: List[float], max_iterations: int, trials_count: int = 50):
        def objective(trial: optuna.Trial) -> int:
            trial_params = [
                trial.suggest_float(name, low, high, log=False)
                for name, (low, high) in params.items()
            ]
            self._f.clear()
            gradient.clear()
            algo_instance = self._algo_partial(lrs(*trial_params))
            algo_instance.find_min(start, max_iterations)
            return len(algo_instance.get_path())

        study = optuna.create_study(study_name="param optimization", direction="minimize")
        study.optimize(objective, n_trials=trials_count)
        print("Количество завершенных испытаний: ", len(study.trials))
        print("Лучшее испытание:")
        result_trial = study.best_trial

        print("  Значение метрики (steps_count): ", result_trial.value)
        print("  Лучшие гиперпараметры: ")
        for key, value in result_trial.params.items():
            print(f"    {key}: {value}")
        return study.best_params
