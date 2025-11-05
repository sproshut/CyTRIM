---
theme: dracula
mdc: true
lineNumbers: true
duration: 15min
defaults:
  preload: true
  transition: slide-left
addons:
  - slidev-component-progress
  - window-mockup

title: Introduction to Numba
author: Proshutinskiy Sergey
transition: slide-left
hideInToc: true
---

# Numba Introduction
<!-- TODO: Add TUW logo in bottom right corner -->

---
---

# Themen

- Was Numba kann
- Einführung in JIT-Kompilierung
- Tests
- Verbesserungspotenzial
- Persönliche Meinung & Fazit

---
layout: two-cols-header
---

# Was Numba kann

::left::

## Kompilierung vom Python-Code

- Optimierung der Rechenoperationen durch CPU-spezifische Anweisungen
- Unterstützt NumPy und meist verwendeten Python-libraries "out of the box"
- Minimale Code-Änderungen notwendig

## Weitere Features

- Parallelisierung der Rechenoperationen / Schleifen
- Ermöglicht Einbindung vom Python-Code in C/C++

::right::

```python {all|1,6}
from numba import jit
import numpy as np

x = np.arange(100).reshape(10, 10)

@jit
def go_fast(a):
    trace = 0.0
    # Numba likes loops
    for i in range(a.shape[0]):
        # Numba likes NumPy functions
        trace += np.tanh(a[i, i])
    # Numba likes NumPy broadcasting
    return a + trace

print(go_fast(x))
```

<style>
.two-cols-header {
  column-gap: 20px; /* Adjust the gap size as needed */
}
</style>

---
---

# JIT-Kompilierung

```python {*}{lines:false} window
# JIT (Just-In-Time) Kompilierung wandelt gekennzeichnete Code-Abschnitte
# zum Zeitpunkt der ersten Ausführung in Maschinencode um, um diese
# bei eineutem Aufruf wiederverwenden zu können
```

> "Numba reads the Python bytecode for a decorated function and combines this with information about the types of the input arguments to the function. It analyzes and optimizes your code, and finally uses the LLVM compiler library to generate a machine code version of your function, tailored to your CPU capabilities. This compiled version is then used every time your function is called."
- Kommt nur in interpretierbaren Programmiersprachen (e.g. PHP) zum Einsatz
- Erste Ausführung dauert länger
- Kompilierte Teile können auf der Festplatte gespeichert und wiederverwendet werden

---
layout: two-cols-header
---

# Tests - Ausgangsbedingungen

::left::

::right::

## Technische Daten

- Intel Core i5-12500H (x86_64)
  - 12 cores, 16 threads
- 24GB DDR4 RAM
- numpy 2.0.2
- numba 0.60.0

---
layout: image
image: media/flow_crabviz.svg
backgroundSize: 70em 70%
---

# Tests - Vorbereitung

---
---

# H1

<WindowMockup codeblock>
```shell
$ echo "Hello, World!"
Hello, World!
```
</WindowMockup>

```python window
print("Hello world!)
```

````md magic-move
```shell {all|1|2}
$ echo "Hello, World!"
Hello, World!
```

```python
print("Hello world!)
```
````

---
layout: intro
---

# Thx
