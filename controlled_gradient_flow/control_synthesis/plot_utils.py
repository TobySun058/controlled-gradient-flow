import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA


def plot_metrics(logs, log_scale=False, figsize=(8, 6), dpi=125):
    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi, sharex=True, constrained_layout=True)
    ax = axes[0]

    ax.plot(list(range(len(logs["||x_target-x_{iter}(T)||2"]))), logs["||x_target-x_{iter}(T)||2"], linestyle='-',
            linewidth=2)
    ax.set_xlabel("Iterate")
    ax.set_title("Target Error")

    ax.grid()
    if log_scale:
        ax.set_yscale("log")
    ax = axes[1]

    ax.plot(list(range(len(logs["||u(t)||2"]))), logs["||u(t)||2"], linestyle='-', linewidth=2, label="Control energy")
    ax.plot(list(range(len(logs["||u(t)||'2"]))), logs["||u(t)||'2"], linestyle='-', linewidth=2,
            label="Energy certificate")
    ax.legend()
    ax.set_xlabel("Itera")
    ax.set_title("Control energy")

    if log_scale:
        ax.set_yscale("log")
    ax.grid()
    plt.suptitle("Control Synthesis Metrics")
    return plt.gcf(), axes
def plot_pca_trajectories_3d(
        logs, obj, plot_iters, figsize=(8, 8), title="Controlled Trajectories in PCA Space",
        dpi=100, use_svd=True):
    model_params = obj.system
    _A, _B, _C = jnp.array(logs["x_{iter}(T)"]).shape
    flattened_trajectories = jnp.array(logs["x_{iter}(T)"]).reshape(
        -1,
        model_params.state_dim)
    center_pt = jnp.zeros_like(obj.x0).reshape(1, -1)
    nominal_traj = jnp.array(logs["nominal_trajectory"])
    pca_fit_coordinates = jnp.vstack(
        [flattened_trajectories, center_pt,
         obj.x0.reshape(1, -1),
         obj.x1.reshape(1, -1), nominal_traj])
    if use_svd:
        pca = TruncatedSVD(n_components=3)
    else:

        pca = PCA(n_components=3)
    all_pcs = pca.fit_transform(pca_fit_coordinates)
    xs_pca = pca.transform(flattened_trajectories).reshape(_A, _B, 3)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pca.transform(obj.x0.reshape(1, -1))[:, 0],
        pca.transform(obj.x0.reshape(1, -1))[:, 1],
        pca.transform(obj.x0.reshape(1, -1))[:, 2],
        color="green",
        s=100,
        label="Init")
    ax.scatter(
        pca.transform(obj.x1.reshape(1, -1))[:, 0],
        pca.transform(obj.x1.reshape(1, -1))[:, 1],
        pca.transform(obj.x1.reshape(1, -1))[:, 2],
        color="red",
        s=100,
        label="Target",
        marker="x")
    zero_pt = pca.transform(center_pt)
    ax.scatter(
        zero_pt[:, 0],
        zero_pt[:, 1],
        zero_pt[:, 2],
        color="black",
        s=100,
        marker="^",
        label="Fixed Point")
    for i in plot_iters:  # range(xs_pca.shape[0]):
        ax.plot(
            xs_pca[i, :, 0],
            xs_pca[i, :, 1],
            xs_pca[i, :, 2],
            label=f"Iter {i}")

    nominal_pca = pca.transform(nominal_traj)
    ax.plot(
        nominal_pca[:, 0],
        nominal_pca[:, 1],
        nominal_pca[:, 2],
        color="gray",
        linestyle="--",
        label="Nominal Trajectory")
    plt.legend()
    plt.xlim(all_pcs[:, 0].min(), all_pcs[:, 0].max())
    plt.ylim(all_pcs[:, 1].min(), all_pcs[:, 1].max())
    ax.set_zlim(all_pcs[:, 2].min(), all_pcs[:, 2].max())
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    ax.set_zlabel("PC 3")
    plt.title(title)
    return fig


def plot_pca_trajectories_2d(
        logs,
        obj,
        plot_iters,
        xlim, ylim,
        title="Controlled Trajectories in PCA Space",
        figsize=(8, 8),
        dpi=100, use_svd=True):
    model_params = obj.system
    _A, _B, _C = jnp.array(logs["x_{iter}(T)"]).shape
    flattened_trajectories = jnp.array(logs["x_{iter}(T)"]).reshape(
        -1,
        model_params.state_dim)
    center_pt = jnp.zeros_like(obj.x0).reshape(1, -1)
    nominal_traj = jnp.array(logs["nominal_trajectory"])
    pca_fit_coordinates = jnp.vstack(
        [flattened_trajectories, center_pt,
         obj.x0.reshape(1, -1),
         obj.x1.reshape(1, -1), nominal_traj])
    if use_svd:
        pca = TruncatedSVD(n_components=2)
    else:

        pca = PCA(n_components=2)
    all_pcs = pca.fit_transform(pca_fit_coordinates)
    xs_pca = pca.transform(flattened_trajectories).reshape(_A, _B, 2)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(
        pca.transform(obj.x0.reshape(1, -1))[:, 0],
        pca.transform(obj.x0.reshape(1, -1))[:, 1],
        color="green",
        s=100,
        label="Init")
    ax.scatter(
        pca.transform(obj.x1.reshape(1, -1))[:, 0],
        pca.transform(obj.x1.reshape(1, -1))[:, 1],
        color="red",
        s=100,
        label="Target",
        marker="x")
    zero_pt = pca.transform(center_pt)
    ax.scatter(
        zero_pt[:, 0],
        zero_pt[:, 1],
        color="black",
        s=100,
        marker="^",
        label="Trivial equilibrium")
    for i in plot_iters:  # range(xs_pca.shape[0]):
        ax.plot(xs_pca[i, :, 0], xs_pca[i, :, 1], label=f"Iter {i}")

    nominal_pca = pca.transform(nominal_traj)
    ax.plot(
        nominal_pca[:, 0],
        nominal_pca[:, 1],
        color="gray",
        linestyle="--",
        label="Nominal Trajectory")
    plt.legend()
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    plt.title(title)
    return fig, ax


def plot_2d_vector_field(
        obj,
        logs,
        n_pts,
        xlim,
        ylim,
        plot_iters,
        figsize=(8, 8),
        dpi=125,
        alpha=0.85,
        cmap='viridis'):
    system = obj.system
    x_init = obj.x0
    x_goal = obj.x1

    xs = jnp.linspace(xlim[0], xlim[1], n_pts)
    ys = jnp.linspace(ylim[0], ylim[1], n_pts)

    # Use JAX meshgrid
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")  # both (n_pts, n_pts)

    # Stack into points: (n_pts*n_pts, 2)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # Vectorize over points
    vf_all = jax.vmap(lambda x: system.N_fn(0.0, x))(
        points)  # (n_pts*n_pts, 2)

    # Split components and norms, reshape back to grid
    U = vf_all[:, 0].reshape((n_pts, n_pts))
    V = vf_all[:, 1].reshape((n_pts, n_pts))
    C = jnp.linalg.norm(vf_all, axis=-1).reshape((n_pts, n_pts))

    plt.figure(figsize=figsize, dpi=dpi)
    X, Y, U, V = np.array(X), np.array(Y), np.array(U), np.array(V)
    plt.streamplot(
        X,
        Y,
        U,
        V,
        density=2,
        linewidth=1,
        arrowsize=1,
        arrowstyle='->',
        zorder=5,
        color="black")
    plt.pcolormesh(
        X,
        Y,
        C,
        shading='auto',
        alpha=alpha,
        cmap=cmap,
        zorder=1,
        vmin=0)
    for i in plot_iters:
        plt.plot(
            np.array(logs["x_{iter}(T)"][i])[::, 0],
            np.array(logs["x_{iter}(T)"][i])[::, 1],
            '-', label=f'Iter {i}', linewidth=2, alpha=1, markersize=2,
            zorder=30)

    plt.plot(
        np.array(logs["nominal_trajectory"])[::, 0],
        np.array(logs["nominal_trajectory"])[::, 1],
        label='Nominal trajectory',
        linewidth=2,
        zorder=30,
        markersize=2,
        color="yellow",
        linestyle='dashed')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Control from ({obj.x0[0]:.2f},{obj.x0[1]:.2f}) to ({obj.x1[0]:.2f},{obj.x1[1]:.2f}) in time {obj.t1:.2f}")
    plt.xlim([xlim[0], xlim[1]])
    plt.ylim([ylim[0], ylim[1]])
    plt.scatter(
        x_init[0], x_init[1], color='orange', label='Initial State', marker='^',
        s=80, zorder=50)
    plt.scatter(
        x_goal[0], x_goal[1], color='red', label='Target State', s=80,
        zorder=50, marker='x')

    plt.legend(loc='lower right')
    plt.colorbar(shrink=0.2)
    plt.gca().set_aspect('equal', adjustable='box')

    return plt.gcf()


def compare_alignment_metrics(obj, logs_g, logs_ag, figsize=(8, 8), dpi=125):
    model_params = obj.system
    ts_g, ts_ag = logs_g["recorded_ts"], logs_ag["recorded_ts"]
    u_l2_g, u_l2_ag = [jax.vmap(lambda x: jnp.linalg.norm(x) ** 2)(jnp.array(x[-1])) for x in
                       [logs_g["u(t)"], logs_ag["u(t)"]]]
    N_norm_g, N_norm_ag = [
        jax.vmap(
            lambda x: jnp.linalg.norm(
                model_params.N_fn(0, x)) ** 2)(
            jnp.array(logs["x_{iter}(T)"][-1])) for logs in [logs_g, logs_ag]]
    Nts_g, Nts_ag = [jax.vmap(lambda x: model_params.N_fn(0, x))(
        jnp.array(logs["x_{iter}(T)"][-1])) for logs in [logs_g, logs_ag]]
    Bts_g, Bts_ag = [jax.vmap(lambda x: model_params.B_fn(0, x))(
        jnp.array(logs["x_{iter}(T)"][-1])) for logs in [logs_g, logs_ag]]
    BNts_g, BNts_ag = [jax.vmap(lambda Bt, Nt: Bt @ Nt)(
        Nts, Bts) for Nts, Bts in [(Nts_g, Bts_g), (Nts_ag, Bts_ag)]]

    cosine_similarity_g, cosine_similarity_ag = [
        jax.vmap(
            lambda u, BN: jnp.clip(
                jnp.dot(u, BN) /
                (jnp.linalg.norm
                 (u) * jnp.linalg.norm(BN)
                 + 1e-12), -1.0, 1.0))(
            jnp.array(logs["u(t)"][-1]),
            BNts) for logs, BNts in [(logs_g, BNts_g), (logs_ag, BNts_ag)]]

    fig, axes = plt.subplots(
        3, 1, figsize=figsize, dpi=dpi, sharex=True,
        constrained_layout=True)
    ax = axes[0]
    ax.plot(ts_g, u_l2_g, label="Gramian Synthesis", linestyle='-', linewidth=2)
    ax.plot(ts_ag, u_l2_ag, label="Optimal-Gramian Synthesis", linestyle='-', linewidth=2)
    ax.set_title("||u(t)||2")
    ax.legend()
    ax.grid()
    ax = axes[1]
    ax.plot(ts_g, N_norm_g, label="Gramian Synthesis", linestyle='-', linewidth=2)
    ax.plot(ts_ag, N_norm_ag, label="Optimal-Gramian Synthesis", linestyle='-', linewidth=2)
    ax.set_title("||N(xt)||2")
    ax.legend()
    ax.grid()
    ax = axes[2]
    ax.plot(ts_g, cosine_similarity_g, label="Gramian Synthesis", linestyle='-', linewidth=2)
    ax.plot(ts_ag, cosine_similarity_ag, label="Optimal-Gramian Synthesis", linestyle='-', linewidth=2)
    ax.set_xlabel("Time")
    ax.set_title("cosine(N(xt),B(t,x(t))@u(t))")
    ax.legend()
    ax.set_xlim([obj.t0, obj.t1])
    ax.grid()
    plt.suptitle("2D Case")
    return plt.gcf()


def plot_alignment_metrics(obj, logs, plot_iters, figsize=(8, 8), dpi=125):

    model_params = obj.system
    u_l2 = jax.vmap(jax.vmap(lambda x: jnp.linalg.norm(x) ** 2))(
        jnp.array(logs["u(t)"]))

    N_norm = jax.vmap(
        jax.vmap(
            lambda x: jnp.linalg.norm(
                model_params.N_fn(
                    0, x)) ** 2))(
        jnp.array(logs["x_{iter}(T)"]))
    Nts = jax.vmap(jax.vmap(lambda x: model_params.N_fn(0, x)))(
        jnp
        .array(logs["x_{iter}(T)"]))
    Bts = jax.vmap(jax.vmap(lambda x: model_params.B_fn(0, x)))(
        jnp
        .array(logs["x_{iter}(T)"]))

    BNts = jax.vmap(jax.vmap(lambda Bt, Nt: Bt @ Nt))(Nts, Bts)
    cosine_similarity = jax.vmap(
        jax.vmap(
            lambda u, BN: jnp.clip(
                jnp.dot(u, BN) /
                (jnp.linalg.norm
                 (u) * jnp.linalg
                 .norm(BN)
                 + 1e-12), -1.0, 1.0)))(
        jnp.array(logs["u(t)"]),
        BNts)
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, dpi=dpi, sharex=True,
        constrained_layout=True)
    ax = axes[0]
    for i in plot_iters:
        ax.plot(
            logs["recorded_ts"], u_l2[i], label=f"Iteration {i}", alpha=0.8,
            linestyle='-', linewidth=2)
        ax.set_xlabel("Time")
        ax.set_title("||u(t)||2")
    ax.legend()
    ax.grid()
    ax = axes[1]
    for i in plot_iters:
        ax.plot(
            logs["recorded_ts"],
            N_norm[i],
            label=f"Iteration {i}",
            alpha=0.8,
            linestyle='-',
            linewidth=2)
        ax.set_xlabel("Time")
        ax.set_title("||N(xt)||2")
    ax.legend()
    ax.grid()
    ax = axes[2]
    for i in plot_iters:
        ax.plot(
            logs["recorded_ts"],
            cosine_similarity[i],
            label=f"Iteration {i}",
            alpha=0.8,
            linestyle='-',
            linewidth=2)
        ax.set_xlabel("Time")
        ax.set_title("cosine(N(xt),B(t,x(t))@u(t))")
    ax.legend()
    ax.grid()
    plt.suptitle("2D Case")
    return plt.gcf()


def compare_2d_vector_field(
        obj,
        logs_g,
        logs_ag,
        n_pts,
        xlim,
        ylim,
        figsize=(8, 8),
        dpi=125,
        alpha=0.85,
        cmap='viridis'):
    system = obj.system
    x_init = obj.x0
    x_goal = obj.x1

    xs = jnp.linspace(xlim[0], xlim[1], n_pts)
    ys = jnp.linspace(ylim[0], ylim[1], n_pts)

    # Use JAX meshgrid
    X, Y = jnp.meshgrid(xs, ys, indexing="xy")  # both (n_pts, n_pts)

    # Stack into points: (n_pts*n_pts, 2)
    points = jnp.stack([X.ravel(), Y.ravel()], axis=-1)

    # Vectorize over points
    vf_all = jax.vmap(lambda x: system.N_fn(0.0, x))(
        points)  # (n_pts*n_pts, 2)

    # Split components and norms, reshape back to grid
    U = vf_all[:, 0].reshape((n_pts, n_pts))
    V = vf_all[:, 1].reshape((n_pts, n_pts))
    C = jnp.linalg.norm(vf_all, axis=-1).reshape((n_pts, n_pts))

    plt.figure(figsize=figsize, dpi=dpi)
    X, Y, U, V = np.array(X), np.array(Y), np.array(U), np.array(V)
    plt.streamplot(
        X,
        Y,
        U,
        V,
        density=2,
        linewidth=1,
        arrowsize=1,
        arrowstyle='->',
        zorder=5,
        color="black")
    plt.pcolormesh(
        X,
        Y,
        C,
        shading='auto',
        alpha=alpha,
        cmap=cmap,
        zorder=1,
        vmin=0)
        # --- NEW: Optimal-gramian iteration 0 trajectory ---
    x_ag0 = np.array(logs_ag["x_{iter}(T)"][0])
    plt.plot(
        x_ag0[:, 0],
        x_ag0[:, 1],
        ':',
        label='Seed control',
        linewidth=2,
        alpha=0.9,
        zorder=25
    )


    plt.plot(
        np.array(logs_g["x_{iter}(T)"][-1])[::, 0],
        np.array(logs_g["x_{iter}(T)"][-1])[::, 1],
        '-', label=f'Gramian', linewidth=2, alpha=1, markersize=2,
        zorder=30)
    plt.plot(np.array(logs_ag["x_{iter}(T)"][-1])[::, 0],
        np.array(logs_ag["x_{iter}(T)"][-1])[::, 1],
        '--', label=f'Optimal gramian', linewidth=2, alpha=1, markersize=2,
        zorder=30)

    plt.plot(
        np.array(logs_g["nominal_trajectory"])[::, 0],
        np.array(logs_g["nominal_trajectory"])[::, 1],
        label='Nominal trajectory',
        linewidth=2,
        zorder=30,
        markersize=2,
        color="yellow",
        linestyle='dashed')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(f"Control from ({obj.x0[0]:.2f},{obj.x0[1]:.2f}) to ({obj.x1[0]:.2f},{obj.x1[1]:.2f}) in time {obj.t1:.2f}")
    plt.xlim([xlim[0], xlim[1]])
    plt.ylim([ylim[0], ylim[1]])
    plt.scatter(
        x_init[0], x_init[1], color='orange', label='Initial State', marker='^',
        s=80, zorder=50)
    plt.scatter(
        x_goal[0], x_goal[1], color='red', label='Target State', s=80,
        zorder=50, marker='x')

    plt.legend(loc='lower left')  #lower right
    plt.colorbar(shrink=0.2)
    plt.gca().set_aspect('equal', adjustable='box')

    return plt.gcf()