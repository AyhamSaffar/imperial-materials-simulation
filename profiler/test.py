import time
import numpy as np

a = np.full(shape=(100, 100), fill_value=10)

start = time.perf_counter()
for i in range(1_000_000):
    a = a - 0.0001
stop = time.perf_counter()
print(f'no_local_vars: {stop-start:.3f}')

def substraction(array):
    temp_a = array - 0.0001
    return temp_a

a = np.full(shape=(100, 100), fill_value=10)

start = time.perf_counter()
for i in range(1_000_000):
    a = substraction(a)
stop = time.perf_counter()
print(f'with_local_vars: {stop-start:.3f}')