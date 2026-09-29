import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

A = np.array([[3.0, 0.0],
              [0.0, 1.0]])

def f(X):
    return 0.5 * X.T @ A @ X

def grad_f(X):
    return A @ X

# GD
def run_gd(X0, gamma=0.25, N=35):
    X = np.zeros((N + 1, 2))
    X[0] = X0
    for n in range(N):
        X[n + 1] = X[n] - gamma * grad_f(X[n])
    return X

# Momentum GD
def run_momentum(X0, gamma=0.18, beta=0.85, N=35):
    X = np.zeros((N + 1, 2))
    X[0] = X0
    X[1] = X[0] - gamma * grad_f(X[0])
    for n in range(1, N):
        X[n + 1] = X[n] - gamma * grad_f(X[n]) + beta * (X[n] - X[n - 1])
    return X

# === Minimum-energy control for continuous-time system Xdot = -A X + B u ===
# Here we take B = I (2x2), consistent with your earlier implicit choice.
def min_energy_control_u(t, T, X0):
    # Controllability Gramian for diagonal A, B = I:
    # Wc = ∫_0^T e^{-A tau} B B^T e^{-A^T tau} d tau
    # With A = diag(3,1), B=I:
    Wc = np.diag([
        (1.0 - np.exp(-6.0 * T)) / 6.0,  # integral of e^{-6 tau}
        (1.0 - np.exp(-2.0 * T)) / 2.0   # integral of e^{-2 tau}
    ])
    Wc_inv = np.linalg.inv(Wc)

    # Desired final state X(T) = 0
    eTA = np.diag([np.exp(-3.0 * T), np.exp(-1.0 * T)])  # e^{-A T}
    rhs = -eTA @ X0  # (X* - e^{-AT}X0) with X* = 0

    # u(t) = B^T e^{-A^T (T-t)} Wc^{-1} (X* - e^{-AT} X0)
    eTtA = np.diag([np.exp(-3.0 * (T - t)), np.exp(-1.0 * (T - t))])
    return eTtA @ Wc_inv @ rhs  # since B = I, B^T does nothing

# === Controlled method: Euler discretization of Xdot = -A X + B u(t) ===
def run_controlled_gd(X0, N=160, T=3.0, B=None):
    if B is None:
        B = np.eye(2)

    X = np.zeros((N + 1, 2))
    X[0] = X0

    ts = np.linspace(0.0, T, N + 1)
    dt = T / N

    for n in range(N):
        u_cont = min_energy_control_u(ts[n], T, X0)   # continuous-time u(t_n)
        X[n + 1] = X[n] + dt * (-A @ X[n] + B @ u_cont)

    return X

# Stochastic GD
def run_stochastic_gd(X0, gamma=0.12, sigma=0.25, N=70, seed=2):
    rng = np.random.default_rng(seed)
    X = np.zeros((N + 1, 2))
    X[0] = X0
    for n in range(N):
        xi = rng.normal(0.0, sigma, size=2)
        X[n + 1] = X[n] - gamma * (grad_f(X[n]) + xi)
    return X

# ==========================
# Run methods
# ==========================
X0 = np.array([3.0, 3.0])
X_gd = run_gd(X0)
X_mom = run_momentum(X0)
X_ctrl = run_controlled_gd(X0)   # updated controlled method
X_sgd = run_stochastic_gd(X0)

# Grid for plotting
x = np.linspace(-4, 4, 250)
y = np.linspace(-4, 4, 250)
Xg, Yg = np.meshgrid(x, y)
Zg = 0.5 * (3.0 * Xg**2 + Yg**2)

def plot_traj_2d(ax, X, label):
    ax.plot(X[:, 0], X[:, 1], linewidth=2, label=label)
    ax.scatter(X[0, 0], X[0, 1], s=30)

def plot_traj_3d(ax, X, label):
    Z = np.array([f(p) for p in X])
    ax.plot(X[:, 0], X[:, 1], Z, linewidth=2, label=label)
    ax.scatter(X[0, 0], X[0, 1], Z[0], s=30)

fig = plt.figure(figsize=(14, 6))

ax2d = fig.add_subplot(1, 2, 1)
ax2d.contourf(Xg, Yg, Zg, levels=35, cmap="inferno")
plot_traj_2d(ax2d, X_gd, "Gradient Descent")
plot_traj_2d(ax2d, X_mom, "Momentum GD")
plot_traj_2d(ax2d, X_ctrl, "Controlled (Euler on Xdot=-AX+Bu)")
plot_traj_2d(ax2d, X_sgd, "Stochastic GD")
ax2d.scatter(0, 0, marker="x", s=80, c="black")
ax2d.set_xlabel("Parameter 1")
ax2d.set_ylabel("Parameter 2")
ax2d.set_title("2D level sets and paths")
ax2d.set_aspect("equal", adjustable="box")
ax2d.legend()

ax3d = fig.add_subplot(1, 2, 2, projection="3d")
ax3d.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.35, linewidth=0)
ax3d.contourf(Xg, Yg, Zg, zdir="z", offset=0.0, levels=35, cmap="inferno", alpha=0.9)

plot_traj_3d(ax3d, X_gd, "Gradient Descent")
plot_traj_3d(ax3d, X_mom, "Momentum GD")
plot_traj_3d(ax3d, X_ctrl, "Controlled (Euler)")
plot_traj_3d(ax3d, X_sgd, "Stochastic GD")

ax3d.scatter(0, 0, 0, marker="x", s=80, c="black")
ax3d.set_xlabel("Parameter 1")
ax3d.set_ylabel("Parameter 2")
ax3d.set_zlabel("f(x,y)")
ax3d.set_title("3D surface and paths")
ax3d.set_zlim(0.0, Zg.max())
ax3d.view_init(elev=30, azim=-55)
ax3d.legend()

plt.tight_layout()
plt.show()
