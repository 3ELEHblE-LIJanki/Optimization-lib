from lrs import gradient

def pretty_print(descent, name, found_result):
    print("""
        {:s}
            found result:         {:f}
            found result in:      {:s}
            steps count:          {:d}
            function calls count: {:d}
            gradient calls count: {:d}    
          """.format(name, 
                     found_result, 
                     str(descent.get_path()[-1]), 
                     len(descent.get_path()), 
                     descent.get_f().get_count(), 
                     gradient.get_count())
    )