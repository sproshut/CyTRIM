# from numba import cfunc, jit
from time import sleep
import numpy as np

# @cfunc("float64(float64, float64)")<
# @jit(cache=True)
def add(x, y):
    return x + y

def main():
    sleep(0.2)
    print(add(5.0, 6.2))
    a = np.random.rand(1000, 2)
    b = np.random.rand(1000000, 2)
    sleep(0.3)

if __name__ == "__main__":
    sleep(0.1)
    main()
    sleep(0.4)