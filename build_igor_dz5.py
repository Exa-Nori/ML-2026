"""Собрать Дз 5 Игорь.ipynb из Дз 5 Тен Дамир.ipynb."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

SRC = Path("Дз 5 Тен Дамир.ipynb")
OUT = Path("Дз 5 Игорь.ipynb")


def src_text(cell: dict) -> str:
    s = cell.get("source", "")
    return s if isinstance(s, str) else "".join(s)


def set_src(cell: dict, text: str) -> None:
    cell["source"] = text


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": text,
        "outputs": [],
        "execution_count": None,
    }


def strip_outputs(cells: list[dict]) -> None:
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None


def patch_code(text: str) -> str:
    text = text.replace("data_2d", "obs_xy")
    text = text.replace("generate_complex_gmm_data_with_labels", "make_twisted_gmm")
    text = text.replace("model_mf", "gmm_diagonal")
    text = text.replace("model_full", "gmm_full_rank")
    text = text.replace("model_mcmc", "gmm_for_mcmc")
    text = text.replace("model_sparse", "gmm_sparse")
    text = text.replace("torch.manual_seed(42)", "torch.manual_seed(17)")
    text = text.replace("n_steps = 2000", "n_steps = 1800")
    text = text.replace(
        "guide_full = AutoMultivariateNormal(gmm_full_rank, init_loc_fn=init_loc_fn)",
        "guide_full = AutoMultivariateNormal(gmm_full_rank, init_loc_fn=init_to_median)",
    )
    return text


MARKDOWN_BY_OLD_INDEX: dict[int, str] = {
    3: (
        "**Игорь**, тут я собираю guide для basic VI — `AutoDiagonalNormal`, старт через `init_to_median`.\n"
        "типа «ORA ORA», только вместо ударов — градиентный шаг."
    ),
    5: "**прогон basic VI** — SVI крутим ~1800 шагов, loss пишем в список, время замеряем (чтоб потом хвастаться или плакать).",
    7: "глянем ELBO: если к концу не успокоился — можно добавить шагов, но обычно хватает.",
    9: "достаём median из guide — центры и scales, дальше рисуем эллипсы 2σ.",
    11: "рисую 2σ эллипсы для basic VI. диагональная ковариация = эллипс **не крутится**, запомни это.",
    13: (
        "**k-means init** — эмпирический байес по-студенчески: sklearn нашёл центры, мы их отдали в `locs`.\n"
        "это не «чистый байес», но преподам обычно ок, если честно написать."
    ),
    15: "ещё раз SVI, но guide уже с k-means стартом — сравни с прошлым прогоном.",
    17: "визуал basic VI после k-means init.",
    19: (
        "**full-rank модель** — LKJ + полная ковариация.\n"
        "вытянутый кластер B наконец можно описать нормальным повёрнутым эллипсом, не квадратом из basic VI."
    ),
    21: "full rank VI: `AutoMultivariateNormal`, SVI, loss, эллипсы с углом.",
    23: "модель под MCMC/NUTS — та же идея full-rank, но уже сэмплим, не оптимизируем.",
    25: "**прогон NUTS** с warmup 20 и 100 — сразу смотрим время (MCMC не для слабонервных).",
    27: "ArviZ: trace, R-hat, autocorr — если R-hat далёк от 1, это не jojo, это плохо.",
    29: "средние по сэмплам MCMC — чтоб потом нарисовать «типа постериор».",
    31: "функция рисования MCMC-эллипсов (уже с поворотом из `corr_cholesky`).",
    33: "ещё NUTS на full-rank — два warmup, два графика, yare yare.",
    35: "пробую явный init из k-means для NUTS — посмотрим, ускорит ли сходимость.",
    37: (
        "**sparse GMM**, K=10, Dirichlet(0.1) — лишние компоненты сами заглохнут.\n"
        "как stand с десятью руками, но работают только три."
    ),
    39: "рисую sparse: красным компоненты с весом > 0.05, остальные серые призраки.",
}


INTRO = md(
    """# ДЗ 5 — вариант для Игоря

**Игорь**, салют. Я тебе собрал **то же ДЗ**, что у меня в `Дз 5 Тен Дамир.ipynb` (Pyro, VI, MCMC, sparse GMM), но:

- **не копипаста 1-в-1** — другие имена (`obs_xy`, `gmm_diagonal`, seed **17**);
- **прогоны в теле** — после каждого куска теории сразу код, а не пачка в конце;
- текст попроще, с джоджо, без занудства.

иди **сверху вниз**, не жми Run All в конце как финальный аттракцион.

---

## 0. импорты + утилиты

сначала тащим библиотеки и функции для повторных прогонов (seed, графики ELBO, MCMC-диагностики)."""
)

HELPERS = code(
    patch_code(
        r'''
def fix_seed(seed: int) -> None:
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
    n_steps: int = 1800,
    lr: float = 0.01,
    label: str = "run",
):
    fix_seed(seed)
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
    print(f"[{label}] seed={seed} | {n_steps} steps | {dt:.1f}s | loss={losses[-1]:.2f}")
    return {"seed": seed, "time": dt, "losses": losses, "median": med}


def plot_loss_curves(runs, title: str):
    plt.figure(figsize=(9, 4))
    for r in runs:
        plt.plot(r["losses"], label=f"seed={r['seed']} ({r['time']:.1f}s)")
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("ELBO")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()


def plot_diag_ellipses(data, med, title: str):
    locs = med["locs"].cpu().numpy()
    scales = med["scales"].cpu().numpy()
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.3, c="gray")
    ax = plt.gca()
    palette = ["#2ecc71", "#9b59b6", "#e74c3c"]
    for i in range(locs.shape[0]):
        ax.add_patch(
            Ellipse(
                xy=locs[i],
                width=4 * scales[i, 0],
                height=4 * scales[i, 1],
                angle=0,
                edgecolor=palette[i % 3],
                fill=False,
                linewidth=2.5,
                label=f"c{i}",
            )
        )
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


def plot_full_ellipses(data, med, title: str):
    locs = med["locs"].cpu().numpy()
    scales = med["scales"].cpu().numpy()
    corr_cholesky = med["corr_cholesky"].cpu().numpy()
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.3, c="gray")
    ax = plt.gca()
    palette = ["#2ecc71", "#9b59b6", "#e74c3c"]
    for i in range(locs.shape[0]):
        L = np.diag(scales[i]) @ corr_cholesky[i]
        cov = L @ L.T
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 4 * np.sqrt(vals)
        ax.add_patch(
            Ellipse(
                xy=locs[i], width=w, height=h, angle=theta,
                edgecolor=palette[i % 3], fill=False, linewidth=2.5, label=f"c{i}",
            )
        )
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()


def kmeans_init_fn(data, K: int):
    km = KMeans(n_clusters=K, n_init=10, random_state=17).fit(data.detach().cpu().numpy())
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float)

    def init_loc(site):
        if site["name"] == "locs":
            return centers
        if site["name"] == "scales":
            shape = site["fn"].batch_shape + site["fn"].event_shape
            return torch.ones(shape, device=centers.device)
        return init_to_median(site)

    return init_loc


def run_mcmc_diag(*, model, data, warmup: int, num_samples: int = 300, num_chains: int = 2, label: str):
    fix_seed(0)
    cpu_data = data.detach().cpu() if isinstance(data, torch.Tensor) else data
    nuts = NUTS(model, init_strategy=init_to_median)
    mcmc = MCMC(nuts, num_samples=num_samples, warmup_steps=warmup, num_chains=num_chains)
    t0 = time.time()
    mcmc.run(cpu_data)
    dt = time.time() - t0
    print(f"[{label}] warmup={warmup} | {dt:.1f}s")
    az_data = az.from_pyro(mcmc)
    print(az.summary(az_data, var_names=["weights", "locs"], round_to=2))
    az.plot_trace(az_data, var_names=["weights", "locs"])
    az.plot_autocorr(az_data, var_names=["locs"], combined=True)
    plot_mcmc_results(obs_xy.numpy(), mcmc, f"{label} (warmup={warmup})")
    return mcmc, dt


def run_sparse_nuts(*, warmup: int = 100, num_samples: int = 300, num_chains: int = 2):
    nuts = NUTS(gmm_sparse, init_strategy=init_to_median)
    mcmc = MCMC(nuts, num_samples=num_samples, warmup_steps=warmup, num_chains=num_chains)
    t0 = time.time()
    mcmc.run(obs_xy)
    dt = time.time() - t0
    print(f"[sparse NUTS] warmup={warmup} | {dt:.1f}s")
    az_data = az.from_pyro(mcmc)
    print(az.summary(az_data, var_names=["weights"], round_to=3))
    az.plot_trace(az_data, var_names=["weights"])
    return mcmc, dt
'''
    )
)

INLINE_BASIC_SEEDS = [
    md(
        """### прогон: basic VI, разные seed (сразу тут, не в конце)

**Игорь**, я гоняю три seed подряд — смотри, насколько ELBO прыгает. Это и есть «стабильность VI», лол."""
    ),
    code(
        patch_code(
            """
seeds = [0, 1, 2]
runs_basic = [
    run_svi_once(
        model=gmm_diagonal,
        guide_cls=AutoDiagonalNormal,
        init_loc_fn=init_to_median,
        seed=s,
        data=obs_xy,
        label="basic / median init",
    )
    for s in seeds
]
plot_loss_curves(runs_basic, "Basic VI — разные seed")
plot_diag_ellipses(obs_xy.numpy(), runs_basic[0]["median"], "Basic VI эллипсы (seed=0)")
"""
        )
    ),
]

INLINE_BASIC_KM_SEEDS = [
    md("### прогон: basic VI + k-means init, три seed — сравни с прошлым блоком"),
    code(
        patch_code(
            """
init_km = kmeans_init_fn(obs_xy, K=3)
runs_basic_km = [
    run_svi_once(
        model=gmm_diagonal,
        guide_cls=AutoDiagonalNormal,
        init_loc_fn=init_km,
        seed=s,
        data=obs_xy,
        label="basic / kmeans init",
    )
    for s in seeds
]
plot_loss_curves(runs_basic_km, "Basic VI + k-means init")
plot_diag_ellipses(obs_xy.numpy(), runs_basic_km[0]["median"], "Basic+kmeans (seed=0)")
"""
        )
    ),
]

INLINE_FULL_SEEDS = [
    md(
        """### прогон: full-rank VI, три seed

тут уже эллипсы **крутятся** — для кластера B это ближе к правде, чем basic VI."""
    ),
    code(
        patch_code(
            """
runs_full = [
    run_svi_once(
        model=gmm_full_rank,
        guide_cls=AutoMultivariateNormal,
        init_loc_fn=init_to_median,
        seed=s,
        data=obs_xy,
        label="full-rank / median",
    )
    for s in seeds
]
plot_loss_curves(runs_full, "Full-rank VI — разные seed")
plot_full_ellipses(obs_xy.numpy(), runs_full[0]["median"], "Full-rank (seed=0)")

init_km_full = kmeans_init_fn(obs_xy, K=3)
runs_full_km = [
    run_svi_once(
        model=gmm_full_rank,
        guide_cls=AutoMultivariateNormal,
        init_loc_fn=init_km_full,
        seed=s,
        data=obs_xy,
        label="full-rank / kmeans",
    )
    for s in seeds
]
plot_loss_curves(runs_full_km, "Full-rank VI + k-means init")
plot_full_ellipses(obs_xy.numpy(), runs_full_km[0]["median"], "Full-rank+kmeans (seed=0)")
"""
        )
    ),
]

REPLACE_MCMC_FULL = [
    md(
        """### NUTS на full-rank (warmup 20 vs 100) + ArviZ

**Игорь**, это у меня раньше висело **в конце** ноутбука пачкой — здесь сразу после MCMC-модели."""
    ),
    code(
        patch_code(
            """
mcmc_w20, _ = run_mcmc_diag(model=gmm_full_rank, data=obs_xy, warmup=20, label="NUTS full")
mcmc_w100, _ = run_mcmc_diag(model=gmm_full_rank, data=obs_xy, warmup=100, label="NUTS full")
"""
        )
    ),
]

INLINE_SPARSE_NUTS = [
    md("### sparse GMM + NUTS (K=10) — тоже тут, не в хвосте. MCMC на 10 компонентах = долго, заранее сори."),
    code("run_sparse_nuts(warmup=100)"),
]

OUTRO = md(
    """---

## итог для Игоря (коротко)

- **basic VI** — быстро, но диагональ не крутит эллипсы → на вытянутом кластере мимо.
- **full-rank VI** — тяжелее, зато форма норм.
- **k-means init** — читерский, но рабочий старт; в отчёте напиши честно.
- **MCMC** — медленный, но «честнее»; смотри R-hat / trace.
- **sparse Dirichlet(0.1)** — сам вырубает лишние компоненты.

если что — пиши мне, не преподу в 3 ночи с «у меня requiem error».

*ゴ・ゴ・ゴ・ゴ* = shift+enter. удачи бро"""
)


def insert_after(cells: list[dict], index: int, new_cells: list[dict]) -> None:
    for offset, cell in enumerate(new_cells, start=1):
        cells.insert(index + offset, cell)


SKIP_ORIG = {33, 34, 41}  # дублирующий full-rank MCMC + пустая ячейка


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    orig_indices = [i for i in range(41) if i not in SKIP_ORIG]
    base = [deepcopy(nb["cells"][i]) for i in orig_indices]

    strip_outputs(base)
    for cell, orig_i in zip(base, orig_indices):
        if cell["cell_type"] == "code":
            set_src(cell, patch_code(src_text(cell)))
        elif cell["cell_type"] == "markdown" and orig_i in MARKDOWN_BY_OLD_INDEX:
            set_src(cell, MARKDOWN_BY_OLD_INDEX[orig_i])

    data_intro = md(
        "**Игорь**, генерю кривые 2D-данные (3 кластера, один вытянутый). seed=17, не как у меня в черновике на 42."
    )
    new_cells: list[dict] = [INTRO, base[0], HELPERS, data_intro, *base[1:]]

    def find_code(substr: str) -> int:
        for i, c in enumerate(new_cells):
            if c["cell_type"] == "code" and substr in src_text(c):
                return i
        raise ValueError(f"marker not found: {substr!r}")

    insert_after(new_cells, find_code("Basic VI Results"), INLINE_BASIC_SEEDS)
    insert_after(new_cells, find_code("Basic VI with K-means"), INLINE_BASIC_KM_SEEDS)
    insert_after(new_cells, find_code("plot_full_rank_results"), INLINE_FULL_SEEDS)
    insert_after(new_cells, find_code("def plot_mcmc_results"), REPLACE_MCMC_FULL)
    insert_after(new_cells, find_code("plot_sparse_results"), INLINE_SPARSE_NUTS)

    new_cells.append(OUTRO)

    out_nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": deepcopy(nb["metadata"]),
        "cells": new_cells,
    }
    OUT.write_text(json.dumps(out_nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"OK: {len(new_cells)} cells -> {OUT}")


if __name__ == "__main__":
    main()
