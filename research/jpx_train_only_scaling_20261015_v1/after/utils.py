import time


def print_running_time(func):
    def inner_func():
        print("\n============================== Program starts running ==============================\n")
        start_time = time.time()
        func()
        end_time = time.time()
        print("\n============================== Program finished running ==============================\n")
        diff = end_time - start_time
        hr = diff // 3600
        min = (diff - hr * 3600) // 60
        sec = diff - hr * 3600 - min * 60
        print(f"Total runtime: {hr}h, {min}m, {sec}s")
    return inner_func
