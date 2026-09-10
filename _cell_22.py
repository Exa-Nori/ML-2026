import torch
import pyro
import pyro.distributions as dist
import time
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoMultivariateNormal
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def model_full(data, K=3):
    D = data.shape[1]
    weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K)))
    
    with pyro.plate("components", K):
        locs = pyro.sample("locs", dist.Normal(torch.zeros(D), 10.0).to_event(1))
        scales = pyro.sample("scales", dist.HalfNormal(torch.ones(D)).to_event(1))
        corr_cholesky = pyro.sample("corr_cholesky", dist.LKJCholesky(D, concentration=torch.tensor(1.0)))
        L_omega = torch.diag_embed(scales) @ corr_cholesky

    mixing_dist = dist.Categorical(weights)
    component_dist = dist.MultivariateNormal(locs, scale_tril=L_omega)
    mixture = dist.MixtureSameFamily(mixing_dist, component_dist)
    
    with pyro.plate("data", len(data)):
        pyro.sample("obs", mixture, obs=data)

# Подготовка обучения
pyro.clear_param_store()
guide_full = AutoMultivariateNormal(model_full, init_loc_fn=init_loc_fn)

adam = pyro.optim.Adam({"lr": 0.01})
svi = SVI(model_full, guide_full, adam, loss=Trace_ELBO())

# Циклы обучения
n_steps = 2000
losses_full = []

print("Начинаем обучение Full Rank VI...")
start_time = time.time()

for step in range(n_steps):
    loss = svi.step(data_2d)
    losses_full.append(loss)
    if step % 500 == 0:
        print(f"Step {step} - Loss: {loss:.2f}")

duration_full = time.time() - start_time
print(f"Обучение завершено за {duration_full:.2f} сек.")

# Визуализация результата
def plot_full_rank_results(data, guide):
    with torch.no_grad():
        medians = guide.median()
        locs = medians['locs'].numpy()
        scales = medians['scales'].numpy()
        corr_cholesky = medians['corr_cholesky'].numpy()
        
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.3, c='gray')
    ax = plt.gca()
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    for i in range(len(locs)):
        # Матрица ковариации Sigma = L 
        L = np.diag(scales[i]) @ corr_cholesky[i]
        cov = L @ L.T
        
        # Форма эллипса
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 4 * np.sqrt(vals) 
        
        el = Ellipse(xy=locs[i], width=width, height=height, angle=theta,
                     edgecolor=colors[i], fill=False, linewidth=3, label=f'Cluster {i}')
        ax.add_patch(el)

    plt.title(f"Full Rank VI Results (Time: {duration_full:.2f}s)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.show()

plot_full_rank_results(data_2d, guide_full)