import numpy as np
import matplotlib.pyplot as plt

class SimplePerceptron:
    def __init__(self, n_inputs, eta=0.12):
        self.weights = np.random.randn(n_inputs + 1) * 0.4
        self.eta = eta

    def _with_bias(self, X):
        X = np.atleast_2d(X)
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def _compute_net(self, X):
        return np.dot(X, self.weights)

    def classify(self, X):
        Xb = self._with_bias(X)
        nets = self._compute_net(Xb)
        return np.where(nets >= 0.0, 1, 0)

    def fit(self, X, targets, max_iter=800):
        Xb = self._with_bias(X)
        targets = np.asarray(targets, dtype=int)
        mistakes_log = []

        for it in range(max_iter):
            wrong = 0

            for x_vec, true_y in zip(Xb, targets):
                y_hat = 1 if self._compute_net(x_vec) >= 0 else 0
                delta = true_y - y_hat

                if delta != 0:
                    self.weights += self.eta * delta * x_vec
                    wrong += 1

            mistakes_log.append(wrong)

            if wrong == 0:
                print(f"Сходимость достигнута на итерации {it+1}")
                break

        return mistakes_log


"""
ИКСЫ И ЕШКИ
"""

points = np.array([[2,4], [-2,4], [2,-4], [-2,-4]])
labels  = np.array([0, 0, 1, 1])

model = SimplePerceptron(n_inputs=2, eta=0.12)
errors_per_epoch = model.fit(points, labels, max_iter=600)


"""
Графики
"""

fig = plt.figure(figsize=(13, 5.2))

"""
график ошибок
"""

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(errors_per_epoch, lw=1.9, color='#2c3e50')
ax1.set_xlabel('Номер итерации')
ax1.set_ylabel('Число ошибок на эпохе')
ax1.grid(ls=':', alpha=0.45)

""" 
Разделяющая поверхность
"""

ax2 = fig.add_subplot(1, 2, 2)

x_min, x_max = -5.2, 5.2
y_min, y_max = -5.2, 5.2

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 280),
                     np.linspace(y_min, y_max, 280))

grid_points = np.c_[xx.ravel(), yy.ravel()]
pred_grid = model.classify(grid_points).reshape(xx.shape)

ax2.contourf(xx, yy, pred_grid,
             levels=[-0.1, 0.5, 1.1],
             colors=['#ffebee', 'white', '#e3f2fd'],
             alpha=0.55)
ax2.contour(xx, yy, pred_grid, levels=[0.5],
            colors='gray', linewidths=1.4, linestyles='--', alpha=0.8) # lw и ls

"""
Точки
"""

cls_colors = ['#c62828' if v==0 else '#1565c0' for v in labels]
ax2.scatter(points[:,0], points[:,1], c=cls_colors,
            s=130, edgecolor='k', lw=1.1, zorder=10)

"""
Разделение
"""

if abs(model.weights[2]) > 1e-9:
    x_vals = np.linspace(x_min, x_max, 180)
    y_vals = - (model.weights[0] + model.weights[1] * x_vals) / model.weights[2]
    ax2.plot(x_vals, y_vals, color='#37474f', lw=2.4,
             label='w1x1 + w2x2 + w0 = 0')

ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_xlabel('Признак x1')
ax2.set_ylabel('Признак x2')
ax2.grid(ls=':', alpha=0.35)
ax2.legend(framealpha=0.92, loc='upper right')
ax2.set_title('Разделяющая прямая')

plt.tight_layout()
plt.show()



print("\nВведите два числа через пробел (x1 x2) или 'q' для выхода\n")

while True:
    s = input(": ").strip()
    if s.lower() in ('q', 'exit', 'выход'):
        break
    try:
        a, b = map(float, s.split())
        res = model.classify([a, b])[0]
        print(f"   : класс {res}  (0 = красный, 1 = синий)")
    except:
        print("   некорректный ввод")