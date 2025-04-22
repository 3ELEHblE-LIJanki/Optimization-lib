import logging
import os
from functools import partial
from typing import List, Type
import optuna
from function_wrapper import FunctionWrapper
from gradient_descent import GradientDecent
from linear_desсent import LinearDecent
from lrs import gradient, LRS
from newton import NewtonCG
import optuna.visualization as vis

ACCEPTABLE_ACCURACY: float = 0.00001
optuna.logging.set_verbosity(logging.CRITICAL)

class ParamOptimizer:
    def __init__(self, algo: Type[GradientDecent | LinearDecent | NewtonCG], f: FunctionWrapper,
                 bounds: List[List[float]], eps: float = ACCEPTABLE_ACCURACY):
        self._f = f
        self._algo_partial = partial(algo, f=self._f, bounds=bounds, eps=eps)
        self.eps = eps

    def optimize(self, lrs: LRS, params: dict[str, tuple[float, float]], start: List[float],
                 max_iterations: int, trials_count: int = 50):
        def objective(trial: optuna.Trial) -> tuple[int, float]:
            try:
                trial_params = []
                for name, ((low, high), param_type) in params.items():
                    if param_type is int:
                        value = trial.suggest_int(name, int(low), int(high), log=True)
                    else:
                        value = trial.suggest_float(name, low, high, log=True)
                    trial_params.append(value)
                self._f.clear()
                gradient.clear()
                algo_instance = self._algo_partial(lrs(*trial_params))
                res = algo_instance.find_min(start, max_iterations)
                return self._f.get_count(), round(res / self.eps) * self.eps
            except Exception as e:
                print(f"Ошибка в испытании: {e}")
                raise optuna.TrialPruned()

        study = optuna.create_study(study_name="param_optimization", directions=["minimize", "minimize"])

        study.optimize(objective, n_trials=trials_count, n_jobs=1)
        fig = vis.plot_pareto_front(study)
        fig.update_layout(
            xaxis_title="Number of Function Evaluations",
            yaxis_title="Function Value (minimized)"
        )
        fig.show()

        if not study.best_trials:
            raise ValueError("No valid trials completed.")

        best_trial = min(study.best_trials, key=lambda trial: trial.values[1])

        print("Search Results:")
        print(f"  Minimal Function Value: {best_trial.values[1]}")
        print(f"  Number of Function Evaluations: {best_trial.values[0]}")
        print("  Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        return best_trial