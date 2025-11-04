import time
import numpy as np

n = 10e6
nums = np.arange(n)
start_time = time.time()
nums = nums**2
end_time = time.time()
print(f"Simulation time: {end_time - start_time:.4f} seconds")