import numpy as np
import matplotlib.pyplot as plt


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        return v / (np.sqrt(np.sum(v * v)) + eps)
    return v / (np.sqrt(np.sum(v * v, axis=1, keepdims=True)) + eps)


def eta_schedule(t, T, eta0=0.28, eta_end=0.02):
    a = t / max(1, T - 1)
    return eta0 * (1 - a) + eta_end * a


def winner(outputs, rng):
    m = outputs.max()
    idx = np.where(np.isclose(outputs, m))[0]
    return int(rng.choice(idx))


def neighborhood_box(M, win, radius):
    i = np.arange(M)
    return (np.abs(i - win) <= radius).astype(np.float32)


def train_som(
    M=16,
    steps=60000,
    low=-2.0,
    high=2.0,
    mode="wta",
    normalize_data=True,
    eta0=0.28,
    eta_end=0.02,
    radius0=3,
    radius_end=1,
    seed=7,
):
    rng = np.random.default_rng(seed)

    # wagi startowe
    W = rng.uniform(low, high, size=(M, 2)).astype(np.float32)
    if normalize_data:
        W = normalize(W)

    for t in range(steps):
        x = rng.uniform(low, high, size=2).astype(np.float32)

        if normalize_data:
            if float(np.linalg.norm(x)) < 1e-12:
                continue
            x_n = normalize(x)
            W_n = normalize(W)
        else:
            x_n = x
            W_n = W

        # wyjścia i winner
        outputs = W_n @ x_n
        win = winner(outputs, rng)

        eta = eta_schedule(t, steps, eta0, eta_end)

        if mode == "wta":
            W[win] += eta * (x_n - W[win])
        else:
            a = t / max(1, steps - 1)
            radius = int(round(radius0 * (1 - a) + radius_end * a))
            N = neighborhood_box(M, win, radius)
            W += (eta * N[:, None]) * (x_n - W)

        if normalize_data:
            W = normalize(W)

    return W


def build_winner_map(
    W,
    grid_n=450,
    low=-2.0,
    high=2.0,
    normalize_data=True,
    chunk_size=200000,
):
    xs = np.linspace(low, high, grid_n, dtype=np.float32)
    ys = np.linspace(low, high, grid_n, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)

    if normalize_data:
        n = np.sqrt(np.sum(pts * pts, axis=1))
        pts_n = np.zeros_like(pts)
        mask = n > 1e-12
        pts_n[mask] = pts[mask] / n[mask, None]
        W_n = normalize(W)
    else:
        pts_n = pts
        W_n = W.astype(np.float32)

    winners = np.empty((pts_n.shape[0],), dtype=np.int32)

    for start in range(0, pts_n.shape[0], chunk_size):
        end = min(start + chunk_size, pts_n.shape[0])
        scores = pts_n[start:end] @ W_n.T
        winners[start:end] = np.argmax(scores, axis=1)

    return winners.reshape((grid_n, grid_n))


def plot_result(grid, W, low=-2.0, high=2.0, title="SOM", save_path=None):
    plt.figure(figsize=(7, 7))
    plt.imshow(
        grid,
        origin="lower",
        extent=[low, high, low, high],
        interpolation="nearest",
    )
    plt.scatter(W[:, 0], W[:, 1], s=55, edgecolors="black")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()


if __name__ == "__main__":

    M = 16
    steps = 60000
    grid_n = 450

    cases = [
        ("WTA | norm=True",  "wta", True),
        ("WTM | norm=True",  "wtm", True),
        ("WTA | norm=False", "wta", False),
        ("WTM | norm=False", "wtm", False),
    ]

    for title, mode, norm in cases:
        W = train_som(
            M=M,
            steps=steps,
            mode=mode,
            normalize_data=norm,
            eta0=0.28,
            eta_end=0.02,
            radius0=3,
            radius_end=1,
            seed=7,
        )

        grid = build_winner_map(
            W,
            grid_n=grid_n,
            normalize_data=norm,
            chunk_size=200000,
        )

        plot_result(
            grid,
            W,
            title=f"SOM | {title} | M={M} | steps={steps}",
            save_path=None,  # np. f"wynik_{mode}_norm{norm}.png"
        )
