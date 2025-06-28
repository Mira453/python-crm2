import numpy as np
import matplotlib.pyplot as plt
from math import sin, pi

# Функція
def f(x):
    return np.sin(x)

# Інтерполяція Лагранжа
def lagrange(x_nodes, y_nodes, x):
    result = 0
    n = len(x_nodes)
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if i != j:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result

# Інтерполяція Ньютона
def newton(x_nodes, y_nodes, x):
    n = len(x_nodes)
    diff = np.zeros((n, n))
    diff[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = (diff[i+1][j-1] - diff[i][j-1]) / (x_nodes[i + j] - x_nodes[i])
    result = diff[0, 0]
    product = 1
    for i in range(1, n):
        product *= (x - x_nodes[i-1])
        result += diff[0, i] * product
    return result

# Точка перевірки
x_star = pi / 4
true_y = sin(x_star)

# Два діапазони
range1 = [0.1*pi, 0.2*pi, 0.3*pi, 0.4*pi]
range2 = [0.1*pi, pi/6, 0.3*pi, 0.4*pi]

ranges = {
    "Діапазон 1": range1,
    "Діапазон 2": range2
}

# Побудова графіків
x_vals = np.linspace(0.1*pi, 0.4*pi, 400)
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

for idx, (label, x_nodes) in enumerate(ranges.items()):
    y_nodes = [f(x) for x in x_nodes]

    # Обчислення значень інтерполяцій
    y_lagrange = [lagrange(x_nodes, y_nodes, x) for x in x_vals]
    y_newton = [newton(x_nodes, y_nodes, x) for x in x_vals]
    y_true = [f(x) for x in x_vals]

    # Похибки в X*
    y_lagr_star = lagrange(x_nodes, y_nodes, x_star)
    y_newton_star = newton(x_nodes, y_nodes, x_star)

    err_lagr = abs(true_y - y_lagr_star)
    err_newton = abs(true_y - y_newton_star)

    # Вивід у консоль
    print(f"\n{label}")
    print(f"  Лагранж: {y_lagr_star:.6f}, похибка: {err_lagr:.6e}")
    print(f"  Ньютон:  {y_newton_star:.6f}, похибка: {err_newton:.6e}")
    print(f"  Точне значення: {true_y:.6f}")

    # Графіки
    ax = axs[idx]
    ax.plot(x_vals, y_true, label='sin(x)', color='black', linestyle='--')
    ax.plot(x_vals, y_lagrange, label='Лагранж', color='blue', linewidth=2, alpha=0.8)
    ax.plot(x_vals, y_newton, label='Ньютон', color='green', linestyle='dashdot', linewidth=2, alpha=0.8)
    ax.scatter(x_nodes, y_nodes, color='red', label='Вузли', zorder=5)
    ax.scatter([x_star], [true_y], color='purple', label=r'$x^* = \pi/4$', zorder=6)
    ax.set_title(label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

