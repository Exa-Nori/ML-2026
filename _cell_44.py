def set_all_seeds(seed: int):
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_svi_once(
    *,
    model,
    guide_cls,
    init_loc_fn,
    seed: int,
    data,
    n_steps: int = 2000,
    lr: float = 0.01,
    label: str = "run",
):
    """Запуск одного SVI-прогона + возврат loss, времени и median-параметров."""
    set_all_seeds(seed)
    pyro.clear_param_store()

    guide = guide_cls(model, init_loc_fn=init_loc_fn)
    svi = SVI(model, guide, pyro.optim.Adam({"lr": lr}), loss=Trace_ELBO())

    losses = []
    t0 = time.time()
    for _ in range(n_steps):
        losses.append(svi.step(data))
    dt = time.time() - t0

    with torch.no_grad():
        med = guide.median()

    print(f"[{label}] seed={seed} | steps={n_steps} | time={dt:.2f}s | final_loss={losses[-1]:.2f}")
    return {
        "seed": seed,
        "time": dt,
        "losses": losses,
        "median": med,
    }


def plot_loss_curves(runs, title: str):
    plt.figure(figsize=(9, 4))
    for r in runs:
        plt.plot(r["losses"], label=f"seed={r['seed']} ({r['time']:.1f}s)")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("ELBO loss")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()


def plot_diag_ellipses_from_median(data, med, title: str):
    """Для Basic VI: эллипсы без поворота (diag scales)."""
    with torch.no_grad():
        locs = med["locs"].cpu().numpy()
        scales = med["scales"].cpu().numpy()

    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.3, c="gray")
    ax = plt.gca()
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    for i in range(locs.shape[0]):
        el = Ellipse(
            xy=locs[i],
            width=4 * scales[i, 0],
            height=4 * scales[i, 1],
            angle=0,
            edgecolor=colors[i % len(colors)],
            fill=False,
            linewidth=3,
            label=f"Cluster {i}",
        )
        ax.add_patch(el)

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


def plot_full_ellipses_from_median(data, med, title: str):
    """Для Full Rank VI: эллипсы с поворотом (ковариация из L @ L.T)."""
    locs = med["locs"].cpu().numpy()
    scales = med["scales"].cpu().numpy()
    corr_cholesky = med["corr_cholesky"].cpu().numpy()

    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.3, c="gray")
    ax = plt.gca()
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    for i in range(locs.shape[0]):
        L = np.diag(scales[i]) @ corr_cholesky[i]
        cov = L @ L.T
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 4 * np.sqrt(vals)

        el = Ellipse(
            xy=locs[i],
            width=width,
            height=height,
            angle=theta,
            edgecolor=colors[i % len(colors)],
            fill=False,
            linewidth=3,
            label=f"Cluster {i}",
        )
        ax.add_patch(el)

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()



def make_kmeans_init_fn(data, K: int):
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(data.detach().cpu().numpy())
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float)

    def init_loc_fn(site):
        if site["name"] == "locs":
            return centers
        if site["name"] == "scales":
            shape = site["fn"].batch_shape + site["fn"].event_shape
            return torch.ones(shape, device=centers.device)
        return init_to_median(site)

    return init_loc_fn



basic_seeds = [0, 1, 2]

runs_basic_default = [
    run_svi_once(
        model=model_mf,
        guide_cls=AutoDiagonalNormal,
        init_loc_fn=init_to_median,
        seed=s,
        data=data_2d,
        label="Basic VI (init_to_median)",
    )
    for s in basic_seeds
]
plot_loss_curves(runs_basic_default, "Basic VI (AutoDiagonalNormal) разные seed (init_to_median)")
plot_diag_ellipses_from_median(data_2d.numpy(), runs_basic_default[0]["median"], "Basic VI эллипсы (пример: seed=0)")

init_loc_fn_km_basic = make_kmeans_init_fn(data_2d, K=3)

runs_basic_km = [
    run_svi_once(
        model=model_mf,
        guide_cls=AutoDiagonalNormal,
        init_loc_fn=init_loc_fn_km_basic,
        seed=s,
        data=data_2d,
        label="Basic VI (KMeans init)",
    )
    for s in basic_seeds
]
plot_loss_curves(runs_basic_km, "Basic VI (AutoDiagonalNormal) разные seed (KMeans init)")
plot_diag_ellipses_from_median(data_2d.numpy(), runs_basic_km[0]["median"], "Basic VI + KMeans init эллипсы (пример: seed=0)")


full_seeds = [0, 1, 2]

runs_full_default = [
    run_svi_once(
        model=model_full,
        guide_cls=AutoMultivariateNormal,
        init_loc_fn=init_to_median,
        seed=s,
        data=data_2d,
        label="Full Rank VI (init_to_median)",
    )
    for s in full_seeds
]
plot_loss_curves(runs_full_default, "Full Rank VI (AutoMultivariateNormal) разные seed (init_to_median)")
plot_full_ellipses_from_median(data_2d.numpy(), runs_full_default[0]["median"], "Full Rank VI — эллипсы (пример: seed=0)")

init_loc_fn_km_full = make_kmeans_init_fn(data_2d, K=3)

runs_full_km = [
    run_svi_once(
        model=model_full,
        guide_cls=AutoMultivariateNormal,
        init_loc_fn=init_loc_fn_km_full,
        seed=s,
        data=data_2d,
        label="Full Rank VI (KMeans init)",
    )
    for s in full_seeds
]
plot_loss_curves(runs_full_km, "Full Rank VI (AutoMultivariateNormal) разные seed (KMeans init)")
plot_full_ellipses_from_median(data_2d.numpy(), runs_full_km[0]["median"], "Full Rank VI + KMeans init эллипсы (пример: seed=0)")