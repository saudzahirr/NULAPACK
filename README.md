# NULAPACK: NUmerical Linear Algebra PACKage

[![Tests](https://github.com/nulapack/nulapack/actions/workflows/test.yml/badge.svg)](https://github.com/nulapack/nulapack/actions/workflows/test.yml)
[![Documentation](https://github.com/nulapack/nulapack/actions/workflows/docs.yml/badge.svg)](https://github.com/nulapack/nulapack/actions/workflows/docs.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![codecov](https://codecov.io/gh/nulapack/nulapack/branch/master/graph/badge.svg)](https://codecov.io/gh/nulapack/nulapack)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=NULAPACK_NULAPACK&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=NULAPACK_NULAPACK)
[![License](https://img.shields.io/badge/license-GPL%20v3-blue.svg)](./LICENSE)

[![PyPI Downloads](https://img.shields.io/pypi/dm/nulapack.svg?label=PyPI%20downloads)](https://pypi.org/project/nulapack/)
[![Python versions](https://img.shields.io/pypi/pyversions/nulapack.svg)](https://pypi.org/project/nulapack/)

NULAPACK is a lightweight, high-performance numerical linear algebra library. It provides a set of core subroutines implemented in Fortran for efficiency, with convenient C++ and Python interfaces.

## Installation

`nulapack` can be installed from **PyPI** or built **from source**.
Using **uv** is recommended for fast and reproducible environments.

---

### Install from PyPI

**Using `pip`**

System-wide (user install):

```bash
pip install --upgrade --user nulapack
```

You may need to use `pip3` instead of `pip` depending on your Python installation.

Inside a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade nulapack
```

---

**Using `uv` (recommended)**

Install into a new environment:

```bash
uv venv
source .venv/bin/activate
uv pip install nulapack
```

Add to an existing `uv` project:

```bash
uv add nulapack
uv sync
```

---

**Using `pipenv`**

```bash
pipenv install nulapack
```

---

**Using `poetry`**

```bash
poetry add nulapack
```

---

**Using `hatch`**

```bash
hatch add nulapack
```

---

### Declaring as a Dependency

`pyproject.toml`:

```toml
[project.dependencies]
nulapack = ">=0.1.0"
```

`requirements.txt`:

```text
nulapack>=0.1.0
```

---

### Building from Source

Building from source is useful if you want the latest features or need to modify the Fortran core.

**Prerequisites**

* CMake ≥ 3.10
* Fortran compiler (e.g. `gfortran`)
* C/C++ compiler (e.g. `gcc`, `clang`)
* Python ≥ 3.9
* uv

**Build and install**

```bash
git clone https://github.com/NULAPACK/NULAPACK.git
cd NULAPACK

uv venv
source .venv/bin/activate

uv run bin/build.py develop --with-py
```

---

## Usage

**Python**

```python
import numpy as np
from nulapack import doolittle

A = np.array([[4, 3], [6, 3]], dtype=np.float64)
L, U, info = doolittle(A)

if info == 0:
    print("L:\n", L)
    print("U:\n", U)
```

**C++**

```cpp
#include <iostream>
#include <vector>
#include "Doolittle.h"

int main() {
    int n = 2;
    std::vector<double> a = {4, 3, 6, 3};
    std::vector<double> l(n * n, 0.0);
    std::vector<double> u(n * n, 0.0);
    int info;

    doolittle(&n, a.data(), l.data(), u.data(), &info);

    if (info == 0) {
        // Use l and u
    }
    return 0;
}
```
