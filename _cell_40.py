def plot_sparse_results(data, guide):
    with torch.no_grad():
        medians = guide.median()
        locs = medians['locs'].numpy()
        scales = medians['scales'].numpy()
        corr_cholesky = medians['corr_cholesky'].numpy()
        weights = medians['weights'].numpy()
        
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=0.2, c='gray')
    ax = plt.gca()
    
    for i in range(len(weights)):
        L = np.diag(scales[i]) @ corr_cholesky[i]
        cov = L @ L.T
        vals, vecs = np.linalg.eigh(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 4 * np.sqrt(vals)
        
        color = 'red' if weights[i] > 0.05 else 'lightgray'
        alpha = 0.8 if weights[i] > 0.05 else 0.3
        
        el = Ellipse(xy=locs[i], width=width, height=height, angle=theta,
                     edgecolor=color, fill=False, linewidth=2, alpha=alpha)
        ax.add_patch(el)
    plt.title(f"Sparse Dirichlet Clustering (K=10, alpha=0.1). Weights: {np.round(weights, 2)}")
    plt.show()

plot_sparse_results(data_2d, guide_sparse)