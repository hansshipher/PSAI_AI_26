import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [ 2,  6],
    [-2,  6],
    [ 2, -6],
    [-2, -6]
], dtype=float)

E = np.array([0, 1, 0, 0], dtype=float)

X1_MIN, X1_MAX = -3.0, 3.0
X2_MIN, X2_MAX = -7.0, 7.0

def step(u: float) -> int:
    return 1 if u >= 0 else 0

def forward_linear(X, w, b):
    return X @ w + b

def forward_class(X, w, b):
    u = forward_linear(X, w, b)
    return np.array([step(val) for val in u], dtype=int)

def train_mse_lms(X, E, lr=0.05, epochs=200, w0=None, b0=0.0):
    n_features = X.shape[1]
    w = np.zeros(n_features, dtype=float) if w0 is None else np.array(w0, dtype=float).copy()
    b = float(b0)
    mse_history = []

    for _ in range(epochs):
        y_lin = forward_linear(X, w, b)
        err = E - y_lin
        mse = np.mean(err ** 2)
        mse_history.append(mse)

        grad_w = -(2.0 / X.shape[0]) * (X.T @ err)
        grad_b = -(2.0 / X.shape[0]) * np.sum(err)

        w = w - lr * grad_w
        b = b - lr * grad_b

    return w, b, np.array(mse_history)

def plot_mse(histories, lrs):
    plt.figure()
    for lr, h in zip(lrs, histories):
        plt.plot(np.arange(1, len(h) + 1), h, label=f"lr={lr}")
    plt.title("MSE vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.legend()

def plot_data_and_boundary(X, E, w, b):
    plt.figure()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

    X0 = X[E == 0]
    X1c = X[E == 1]

    if len(X0) > 0:
        plt.scatter(X0[:, 0], X0[:, 1], marker="o", label="e=0")
    if len(X1c) > 0:
        plt.scatter(X1c[:, 0], X1c[:, 1], marker="s", label="e=1")

    w1, w2 = w[0], w[1]

    if abs(w2) >= 1e-12:
        x1_vals = np.linspace(X1_MIN, X1_MAX, 200)
        x2_vals = -(w1 * x1_vals + b) / w2
        plt.plot(x1_vals, x2_vals, label="boundary")
    elif abs(w1) >= 1e-12:
        x1_const = -b / w1
        plt.axvline(x=x1_const, label="boundary")

    plt.xlim(X1_MIN, X1_MAX)
    plt.ylim(X2_MIN, X2_MAX)
    plt.legend()

def interactive_mode(w, b):
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(True)
    ax.set_xlim(X1_MIN, X1_MAX)
    ax.set_ylim(X2_MIN, X2_MAX)

    X0 = X[E == 0]
    X1c = X[E == 1]

    if len(X0) > 0:
        ax.scatter(X0[:, 0], X0[:, 1], marker="o", label="train e=0")
    if len(X1c) > 0:
        ax.scatter(X1c[:, 0], X1c[:, 1], marker="s", label="train e=1")

    w1, w2 = w[0], w[1]

    if abs(w2) >= 1e-12:
        x1_vals = np.linspace(X1_MIN, X1_MAX, 200)
        x2_vals = -(w1 * x1_vals + b) / w2
        ax.plot(x1_vals, x2_vals, label="boundary")
    elif abs(w1) >= 1e-12:
        x1_const = -b / w1
        ax.axvline(x=x1_const, label="boundary")

    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()

    while True:
        s = input("x1 x2 (or q): ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            parts = s.replace(",", " ").split()
            if len(parts) != 2:
                continue
            x1v = float(parts[0])
            x2v = float(parts[1])
        except:
            continue

        u = x1v * w[0] + x2v * w[1] + b
        cls = step(u)

        marker = "x" if cls == 0 else "*"
        ax.scatter([x1v], [x2v], marker=marker)
        fig.canvas.draw()
        fig.canvas.flush_events()

        print(f"class = {cls}")

    plt.ioff()
    plt.show()

def main():
    lrs = [0.01, 0.05, 0.1]
    epochs = 200

    histories = []
    last_w, last_b = None, None

    w_init = np.array([0.0, 0.0])
    b_init = 0.0

    for lr in lrs:
        w, b, mse_hist = train_mse_lms(X, E, lr=lr, epochs=epochs, w0=w_init, b0=b_init)
        histories.append(mse_hist)
        last_w, last_b = w, b

        y_pred = forward_class(X, w, b)
        acc = np.mean(y_pred == E.astype(int))
        print(f"lr={lr}: w={w}, b={b:.4f}, acc={acc:.2f}")

    plot_mse(histories, lrs)
    plot_data_and_boundary(X, E, last_w, last_b)
    plt.show()

    interactive_mode(last_w, last_b)

if __name__ == "__main__":
    main()