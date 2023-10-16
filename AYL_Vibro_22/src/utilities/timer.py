from time import time

def time_perf(func): 
    def wrap_func(*args, **kwargs):
        start_time = time()*1000
        result = func(*args, **kwargs)
        stop_time = time()*1000

        print(f"{' '*10}Function {func.__name__} executed in {(stop_time-start_time):.4f} milliseconds.")
        return result
    return wrap_func