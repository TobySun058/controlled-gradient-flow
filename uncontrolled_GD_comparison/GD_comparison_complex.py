import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# -----------------------------
# Problem setup: soft-min of 3 quadratics
# -----------------------------
centers = np.array([
    [-2.0, -1.0],
    [ 2.2,  0.5],
    [ 0.0,  2.4]
], dtype=float)

As = np.array([
    [[2.5, 0.0],
     [0.0, 0.8]],

    [[1.2, 0.2],
     [0.2, 2.0]],

    [[0.9, -0.15],
     [-0.15, 1.8]]
], dtype=float)

offsets = np.array([0.0, 0.25, 0.1], dtype=float)
k_softmin = 6.0

def q_i(x, i):
    d = x - centers[i]
    return 0.5 * d.T @ As[i] @ d + offsets[i]

def grad_q_i(x, i):
    return As[i] @ (x - centers[i])

def f(x):
    # softmin of three quadratics
    qs = np.array([q_i(x, i) for i in range(3)])
    m = qs.min()  # numerical stability
    s = np.sum(np.exp(-k_softmin * (qs - m)))
    return m - (1.0 / k_softmin) * np.log(s)

def grad_f(x):
    # gradient of softmin: weighted average of grad(q_i)
    qs = np.array([q_i(x, i) for i in range(3)])
    m = qs.min()
    w_unnorm = np.exp(-k_softmin * (qs - m))
    w = w_unnorm / np.sum(w_unnorm)
    g = np.zeros(2)
    for i in range(3):
        g += w[i] * grad_q_i(x, i)
    return g

# -----------------------------
# Baselines
# -----------------------------
def run_gd(X0, gamma=0.10, N=140):
    X = np.zeros((N + 1, 2))
    X[0] = X0
    for n in range(N):
        X[n + 1] = X[n] - gamma * grad_f(X[n])
    return X

def run_momentum(X0, gamma=0.07, beta=0.90, N=140):
    X = np.zeros((N + 1, 2))
    X[0] = X0
    X[1] = X[0] - gamma * grad_f(X[0])
    for n in range(1, N):
        X[n + 1] = X[n] - gamma * grad_f(X[n]) + beta * (X[n] - X[n - 1])
    return X

def run_stochastic_gd(X0, gamma=0.10, sigma=0.25, N=180, seed=2):
    rng = np.random.default_rng(seed)
    X = np.zeros((N + 1, 2))
    X[0] = X0
    for n in range(N):
        xi = rng.normal(0.0, sigma, size=2)
        X[n + 1] = X[n] - gamma * (grad_f(X[n]) + xi)
    return X

# -----------------------------
# Estimate (approx) global minimizer on a grid (for plotting / target)
# -----------------------------
def estimate_global_min_on_grid(xmin=-4, xmax=4, ymin=-4, ymax=4, n=181):
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    best_val = np.inf
    best_xy = None
    for x in xs:
        for y in ys:
            v = f(np.array([x, y]))
            if v < best_val:
                best_val = v
                best_xy = (x, y)
    return np.array(best_xy), best_val

x_star, f_star = estimate_global_min_on_grid()

# -----------------------------
# "Controlled GD" that matches the SLIDE ASSUMPTION more faithfully:
#   slide assumes continuous linear system:  Xdot = -A X + B u
#   min-energy control uses the controllability Gramian:
#      Wc = ∫_0^T e^{-tA} B B^T e^{-tA^T} dt
#      u(t) = B^T e^{-(T-t)A^T} Wc^{-1} (x_target - e^{-T A} x0)
#
# For our nonlinear f, we use a LOCAL LINEAR PROXY: A ≈ Hessian of f at x_star.
# Then we apply the min-energy law in a receding-horizon way (feedback-ish):
# at each step, treat current state as x0 and remaining time as T_rem.
# -----------------------------

def numerical_hessian_of_f(x, eps=1e-4):
    """
    Hessian H_ij = d/dx_j (grad_f(x))_i via central differences.
    """
    H = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = 1.0
        g_plus  = grad_f(x + eps * e)
        g_minus = grad_f(x - eps * e)
        # column j:
        H[:, j] = (g_plus - g_minus) / (2.0 * eps)
    # symmetrize (Hessian should be symmetric; numerical noise might break it)
    return 0.5 * (H + H.T)

def expm_2x2(M):
    """
    Matrix exponential for 2x2 using eigen-decomposition.
    """
    vals, vecs = np.linalg.eig(M)
    Vinv = np.linalg.inv(vecs)
    return np.real_if_close(vecs @ np.diag(np.exp(vals)) @ Vinv)

def controllability_gramian(A, B, T, n_steps=400):
    """
    Numerically compute Wc = ∫_0^T e^{-tA} B B^T e^{-tA^T} dt by trapezoid rule.
    """
    if T <= 0:
        return np.zeros((2, 2))
    ts = np.linspace(0.0, T, n_steps)
    dt = ts[1] - ts[0]
    W = np.zeros((2, 2))
    BBt = B @ B.T
    for k, t in enumerate(ts):
        E = expm_2x2(-t * A)
        integrand = E @ BBt @ E.T
        w = 0.5 if (k == 0 or k == len(ts) - 1) else 1.0
        W += w * integrand
    return W * dt

def min_energy_u_at_start(A, B, T, x0, x_target, gram_steps=400, ridge=1e-9):
    """
    Compute u(0) for the min-energy control that steers x(0)=x0 to x(T)=x_target
    for the linear system xdot = -A x + B u.

    u(0) = B^T e^{-T A^T} Wc^{-1} (x_target - e^{-T A} x0)
    """
    if T <= 0:
        return np.zeros(B.shape[1])

    Wc = controllability_gramian(A, B, T, n_steps=gram_steps)
    # regularize slightly in case Wc is ill-conditioned
    Wc_reg = Wc + ridge * np.eye(Wc.shape[0])
    Wc_inv = np.linalg.inv(Wc_reg)

    eTA = expm_2x2(-T * A)
    rhs = x_target - eTA @ x0
    u0 = B.T @ expm_2x2(-T * A.T) @ Wc_inv @ rhs
    return np.real_if_close(u0)

def run_controlled_gd(
    X0,
    gamma=0.05,
    N=160,
    T=3.0,
    B=None,
    A_proxy=None,
    gram_steps=400
):
    """
    Discrete controlled GD (Euler discretization of controlled flow):
      X_{n+1} = X_n - gamma * grad_f(X_n) + gamma * B u_n
    where u_n is computed from min-energy control on a linear proxy system.

    - Uses receding horizon: remaining time T_rem = T - t_n
    - Target is x_star (estimated minimizer)
    - Linear proxy A is Hessian of f at x_star unless provided
    """
    if B is None:
        B = np.eye(2)  # k=2 controls

    if A_proxy is None:
        A_proxy = numerical_hessian_of_f(x_star)  # proxy for "A" in slide

    # Ensure shapes
    B = np.asarray(B, dtype=float)
    A_proxy = np.asarray(A_proxy, dtype=float)

    X = np.zeros((N + 1, 2))
    X[0] = X0
    ts = np.linspace(0.0, T, N + 1)

    for n in range(N):
        T_rem = max(T - ts[n], 0.0)

        # Compute u_n from linear proxy to steer from current X[n] to x_star by time T_rem
        u_n = min_energy_u_at_start(
            A=A_proxy,
            B=B,
            T=T_rem,
            x0=X[n],
            x_target=x_star,
            gram_steps=gram_steps
        )

        # Controlled discrete update (Euler):
        X[n + 1] = X[n] - gamma * grad_f(X[n]) + gamma * (B @ u_n)

    return X, A_proxy

# -----------------------------
# Run everything
# -----------------------------
X0 = np.array([2.0, -2.0])

X_gd  = run_gd(X0)
X_mom = run_momentum(X0)
X_sgd = run_stochastic_gd(X0)

X_ctrl, A_used = run_controlled_gd(X0)

print("Estimated minimizer x_star =", x_star, "  f(x_star)≈", f_star)
print("A_proxy used (Hessian at x_star):\n", A_used)

# -----------------------------
# Grid for plots
# -----------------------------
x = np.linspace(-4, 4, 260)
y = np.linspace(-4, 4, 260)
Xg, Yg = np.meshgrid(x, y)
Zg = np.zeros_like(Xg)
for i in range(Xg.shape[0]):
    for j in range(Xg.shape[1]):
        Zg[i, j] = f(np.array([Xg[i, j], Yg[i, j]]))

def plot_traj_2d(ax, X, label):
    ax.plot(X[:, 0], X[:, 1], linewidth=2, label=label)
    ax.scatter(X[0, 0], X[0, 1], s=35)

def plot_traj_3d(ax, X, label):
    Z = np.array([f(p) for p in X])
    ax.plot(X[:, 0], X[:, 1], Z, linewidth=2, label=label)
    ax.scatter(X[0, 0], X[0, 1], Z[0], s=35)

# -----------------------------
# Plot 2D + 3D
# -----------------------------
fig = plt.figure(figsize=(14, 6))

ax2d = fig.add_subplot(1, 2, 1)
ax2d.contourf(Xg, Yg, Zg, levels=40, cmap="inferno")
plot_traj_2d(ax2d, X_gd,   "Gradient Descent")
plot_traj_2d(ax2d, X_mom,  "Momentum GD")
plot_traj_2d(ax2d, X_ctrl, "Controlled GD")
plot_traj_2d(ax2d, X_sgd,  "Stochastic GD")

# Mark estimated global minimizer and parabola centers
ax2d.scatter(x_star[0], x_star[1], marker="x", s=90, c="cyan")
ax2d.scatter(centers[:, 0], centers[:, 1], marker="o", s=45, c="white")

ax2d.set_xlabel("x")
ax2d.set_ylabel("y")
ax2d.set_title("Three-parabola soft-min: level sets and paths")
ax2d.set_aspect("equal", adjustable="box")
ax2d.legend()

ax3d = fig.add_subplot(1, 2, 2, projection="3d")
ax3d.plot_surface(Xg, Yg, Zg, cmap="viridis", alpha=0.35, linewidth=0)
ax3d.contourf(Xg, Yg, Zg, zdir="z", offset=Zg.min(), levels=40, cmap="inferno", alpha=0.9)

plot_traj_3d(ax3d, X_gd,   "Gradient Descent")
plot_traj_3d(ax3d, X_mom,  "Momentum GD")
plot_traj_3d(ax3d, X_ctrl, "Controlled GD")
plot_traj_3d(ax3d, X_sgd,  "Stochastic GD")

ax3d.scatter(x_star[0], x_star[1], f_star, marker="x", s=90, c="cyan")
ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("f(x,y)")
ax3d.set_title("3D surface and paths")
ax3d.set_zlim(Zg.min(), Zg.max())
ax3d.view_init(elev=30, azim=-55)
ax3d.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Objective vs iteration
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot([f(p) for p in X_gd],   label="GD")
plt.plot([f(p) for p in X_mom],  label="Momentum")
plt.plot([f(p) for p in X_ctrl], label="Controlled")
plt.plot([f(p) for p in X_sgd],  label="SGD")
plt.axhline(f_star, linestyle="--", label="estimated min on grid")
plt.xlabel("iteration")
plt.ylabel("f(x)")
plt.title("Objective value vs iteration (three-parabola soft-min)")
plt.legend()
plt.tight_layout()
plt.show()
