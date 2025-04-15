from lrs import gradient, hessian


def pretty_print(descent, name, found_result, gradient=None, hessian=None):
    grad = 0 if gradient is None else gradient.get_count()
    hess = 0 if hessian is None else hessian.get_count()

    print("""
        {:s}
            found result:         {:f}
            found result in:      {:s}
            steps count:          {:d}
            function calls count: {:d}
            gradient calls count: {:d}
            hessian calls count:  {:d}
          """.format(name,
                     found_result,
                     str(descent.get_path()[-1]),
                     len(descent.get_path()),
                     descent.get_f().get_count(),
                     grad,
                     hess)
          )
