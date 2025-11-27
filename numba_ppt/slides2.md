---
theme: dracula
mdc: true
lineNumbers: false
duration: 15min
defaults:
  preload: true
  transition: slide-left
addons:
  - slidev-component-progress
  - window-mockup

title: Optimizing PyTrim (Pt. 2)
author: Proshutinskiy Sergey
transition: slide-left
hideInToc: true
---

# Optimizing PyTrim (Pt. 2)
## Using a combination of NumPy and Numba

---
---

# Content

- Benchmarks
- Implementation
    - Improving performance using pure NumPy
    - Adding Numba
    - Parallelizing
- Conclusion

---
layout: image
image: media/sim_perf.png
backgroundSize: 70%
---

<!--
Performance improvement to relative to original:
- Numba: 36x
- Numpy: 40x
- Numpy + Numba: 78x
-->

---
---

# Improving performance using pure NumPy

- Current version of PyTrim heavily relies on python loops
    - NumPy can't optimize them -> no SIMD optimizations possible
    - Numba can, but has its own limitations (e.g. reduced \[Num\]Py feature support)
<br><br>
### Proposal
- Vectorize everything and rely on more efficient NumPy methods
    - Adding Numba later should be possible and may be beneficial

<!-- SIMD = "Single Instruction Multiple Data" -->

---
layout: two-cols-header
---

# Improving performance using pure NumPy

::left::

- Current state: a maximum of 3 "parallel" arithmetical operations
    - Full simulation for a single ion per iteration
<v-click at="1">

- *Proposal*: as much as possible independent "parallel" arithmetical operations
    - Single simulation step for _all_ particles per iteration
    - Parallelization may be possible / beneficial on larger number of particles

</v-click>

::right::
````md magic-move
```py
import numpy as np

a = np.random.rand(5, 3)
for i in range(5):
    # 3 simultaneous operations
    a[i] += np.random.rand(3)
# Time with 100k rows: 46.48s
print(a)
```

```py
import numpy as np

a = np.random.rand(5, 3)
b = np.random.rand(5, 3)
# 15 simultaneous operations
a += b
# Time with 100k rows: 0.003s
print(a)
```
````

<!-- 100k rows: 46.48s -> 0.003s -->

---
layout: fact
---

# DEMO
### Improving performance using pure NumPy

---
---

# Adding Numba

The following methods / signatures did not compile with Numba:
```py
np.linalg.norm(axis)
np.ndarray[list, list]
np.ndarray[:, list[list]]
np.put_along_axis()
np.argmin(keepdims)
```
(full list can be found <a href="https://numba.readthedocs.io/en/stable/reference/numpysupported.html">here</a>)

Those parts had to be re-written, sometimes in ugly ways

<v-click>

PyQT integration:
**Adding Numba prevents the use of Queues to send simulation updates to UI thread**
</v-click>

---
layout: full
---

# Adding Numba - Parallelization

````md magic-move
```py
def simulate(nion):
    # Initial conditions of the projectile
    e_init = np.full(nion, 50000.0)  # energy (eV)
    pos_init = np.zeros((nion, 3))  # position (A)
    dir_init = np.full((nion, 3), [0.0, 0.0, 1.0])  # direction (unit vector)

    pos, dir, e, is_inside = trajectory.trajectories(pos_init, dir_init, e_init)

    count_inside = np.count_nonzero(is_inside)
    mean_z = pos[is_inside, 2]
    std_z = np.std(mean_z)
    mean_z = np.mean(mean_z)

    return pos, dir, e, is_inside, mean_z, std_z, count_inside
```

```py
@jit(fastmath=True, parallel=True, nogil=True, cache=True)
def simulate(nion: int, nthreads: int = 16):
    # Initial conditions of the projectile
    # NOTE In reality a check is performed to split data in equal parts
    threads = nthreads
    col_cnt = nion / nthreads
    e = np.full((threads, col_cnt), 50000.0)  # energy (eV)
    pos = np.zeros((threads, col_cnt, 3))  # position (A)
    dir = np.zeros((threads, col_cnt, 3))
    dir[:, :, 2] = 1.0  # direction (unit vector)
    is_inside = np.full((threads, col_cnt), True)

    for i in prange(threads):
        trajectory.trajectories(pos[i], dir[i], e[i], is_inside[i])

    pos = pos.reshape((nion, 3))
    dir = dir.reshape((nion, 3))
    e = e.flatten()
    is_inside = is_inside.flatten()
    count_inside = np.count_nonzero(is_inside)
    mean_z = pos[is_inside, 2]
    std_z = np.std(mean_z)
    mean_z = np.mean(mean_z)

    return pos, dir, e, is_inside, mean_z, std_z, count_inside
```
````

---
layout: image
image: media/sim_perf_par.png
backgroundSize: 70%
---

<!-- Parallel performance boost: 247x -->

---
---

# Conclusion

- Provided NumPy implementation brings a significant performance boost
    - Additional 2x possible by reducing precesion from `float64` -> `float32`
- Adding Numba brings a slight improvement, but also significant limitations and less readable code
    - Limited \[Num\]Py feature set
    - Increased codebase size / complexity
- *UI integration options*
    - Vanilla NumPy implementation (in background) notifying about finished simulations (processing in main / other thread)
    - Numba with data splitting, but with Python threads

---
layout: intro
---

# Questions?
