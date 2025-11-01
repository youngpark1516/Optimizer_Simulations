# 🧠 Optimizer Simulations

A lightweight Python project that visualizes how different optimization algorithms behave when minimizing a function — using a custom-built **Polynomial class** as the loss function.  
This repository evolved from a single experimental script into a modular mini-library (`miniopt`) for reusable and extensible optimizer simulations.

---

## 📖 Introduction

The goal of **Optimizer Simulations** is to demonstrate the intuition behind various gradient-based optimization algorithms such as Gradient Descent, AdaGrad, RMSProp, AdaDelta, and Adam.

The project creates a *polynomial function* to act as a loss landscape and visualizes how each optimizer updates its guesses to minimize the function value over iterations.

---

## 🧩 Project Structure

```
Optimizer_Simulations/
│
├── miniopt/                  # Core library (importable as 'miniopt')
│   ├── __init__.py
│   ├── polynomial.py         # Custom polynomial class with differentiation
│   ├── optimizers.py         # GD, AdaGrad, RMSProp, Adam implementations
│   ├── functions.py          # Optional test functions
│   ├── runner.py             # Optimization engine (no visualization)
│   └── viz.py                # Visualization utilities
│
├── examples/                 # Demonstration scripts
│   ├── demo_polynomial.py
│   ├── demo_quartic.py
│   └── compare_optimizers.py
│
├── legacy/                   # Original single-file implementation
│   └── optimizer_v1.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Imports and Dependencies

This project uses three main Python libraries:

### 🧮 NumPy  
A fundamental library for numerical computation, used for:
- Handling coefficients of the polynomial  
- Efficiently evaluating and differentiating polynomial functions  

### 📈 Matplotlib  
A visualization library used for:
- Plotting polynomial curves  
- Animating optimizer trajectories  
- Displaying convergence over iterations  

### 🎲 Random (Standard Library)  
Used for generating random starting points and randomized polynomial coefficients.

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧠 Polynomial Class

The `Polynomial` class defines a simple mathematical function that acts as the *loss function* for optimization.

### Attributes
- `self.coefficients`: NumPy array of coefficients, starting from power \( x^0 \) up to \( x^n \).

### Initialization
You can initialize a polynomial in two ways:
1. **From coefficients:**
   ```python
   p = Polynomial([1, -2, 3])  # 1 - 2x + 3x^2
   ```
2. **Randomized by degree:**
   ```python
   p = Polynomial(degree=3)
   ```

Internally, the class supports:
- `__call__`: evaluate \( f(x) \)
- `derivative()`: return a new `Polynomial` representing \( f'(x) \)
- `domain()`: generate arrays for plotting
- Proper vectorized evaluation for arrays of x-values

---

## ⚡ Optimizers Implemented

Each optimizer updates the parameter \( x \) using its own adaptive learning strategy:

| Optimizer | Description |
|------------|-------------|
| **Gradient Descent (GD)** | Standard first-order method that updates weights opposite to the gradient direction. |
| **AdaGrad** | Adapts learning rate per parameter, scaling inversely with accumulated gradient magnitude. |
| **RMSProp** | Maintains a moving average of squared gradients for smoother adaptive learning rates. |
| **AdaDelta** | An extension of RMSProp that eliminates the need for a manually selected learning rate. |
| **Adam** | Combines momentum (first moment) and RMSProp (second moment) into a highly efficient optimizer. |

Each optimizer class implements:
```python
class OptimizerName:
    def step(self, x, grad):
        ...
```

---

## 🧪 Implementation Overview

The simulation follows these steps:

1. **Define a function** — e.g., a quadratic or cubic polynomial.  
2. **Compute its derivative** — used as the gradient function.  
3. **Choose an optimizer** — e.g., `Adam(lr=0.05)`.  
4. **Run optimization** using `run_1d()` in `miniopt/runner.py`.  
5. **Visualize the results** via `miniopt.viz`.

Example (`examples/demo_polynomial.py`):
```python
from miniopt import Polynomial, Adam, run_1d, viz

p = Polynomial([0, 0, 3])   # f(x) = 3x²
dp = p.derivative()
opt = Adam(lr=0.1)

history = run_1d(f=p, f_prime=dp, optimizer=opt, x0=2.5, steps=50)
viz.plot_path_1d(p, history, x_min=-3, x_max=3, title="Adam on 3x²")
```

---

## 📊 Visualization Features

`miniopt.viz` provides:
- **Function plot:** visualize the shape of the polynomial  
- **Path plot:** show optimizer trajectory step-by-step  
- **Convergence plot:** plot \( f(x_t) \) vs iteration  

---

## Running the Examples

Run examples from the project root:

```bash
python -m examples.demo_polynomial
```

Or compare optimizers:
```bash
python -m examples.compare_optimizers
```

---

## Future Improvements

- Extend to **multi-dimensional optimization**
- Add **momentum**, **NAdam**, and **LAMB** optimizers
- Integrate **interactive visualizations** with Plotly
- Add **unit tests** for optimizer behavior and convergence

---

## 🧾 License

This project is open-source and available under the MIT License.

---

## Author

**Chanyoung Park (Paul / 노유민)**  
Undergraduate Data Science Major at UC San Diego  
*GitHub:* [@ChanyoungPark](https://github.com/ChanyoungPark)
