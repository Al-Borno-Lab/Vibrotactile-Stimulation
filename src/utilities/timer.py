from time import time

def time_perf(func): 
    def wrap_func(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        stop_time = time()

        print(f"    Function {func.__name__} executed in {(stop_time-start_time):.4f} seconds.")
        return result
    return wrap_func