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
                     str([ "{:.20f}".format(x_i) for x_i in descent.get_path()[-1]]),
                     len(descent.get_path()),
                     descent.get_f().get_count(),
                     grad,
                     hess)
          )

def pretty_dataset_print(descent, name, found_result, gradient=None, hessian=None):
    grad = 0 if gradient is None else gradient.get_count()
    hess = 0 if hessian is None else hessian.get_count()

    print("""
        {:s}
            found min error:         {:f}
            found weights:      {:s}
            steps count:          {:d}
            function calls count: {:d}
            gradient calls count: {:d}
          """.format(name,
                     found_result,
                     str([ "{:.20f}".format(x_i) for x_i in descent.get_path()[-1]]),
                     len(descent.get_path()),
                     descent.get_f().get_count(),
                     grad)
          )
    
def pretty_print_annealing(descent, name, found_result):
    found_x = descent.x if descent.x is not None else []
    func_calls = descent.get_f().get_count()
    
    print(f"""
        {name}
            Found result: {found_result}
            Final point: {found_x}
            Alpha: {descent.alpha:.4f}
            T0: {descent.T0:.2f}
            Steps count: {len(descent.get_path())}
            Function calls count: {func_calls}
    """)