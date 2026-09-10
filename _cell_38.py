def model_sparse(data, K=10):
    D = data.shape[1]
    weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K) * 0.1))
    
    with pyro.plate("components", K):
        locs = pyro.sample("locs", dist.Normal(torch.zeros(D), 10.).to_event(1))
        scales = pyro.sample("scales", dist.HalfNormal(torch.ones(D)).to_event(1))
        corr_cholesky = pyro.sample("corr_cholesky", dist.LKJCholesky(D, concentration=1.0))
        L_omega = torch.diag_embed(scales) @ corr_cholesky

    mixture = dist.MixtureSameFamily(dist.Categorical(weights), dist.MultivariateNormal(locs, scale_tril=L_omega))
    with pyro.plate("data", len(data)):
        pyro.sample("obs", mixture, obs=data)

pyro.clear_param_store()
guide_sparse = AutoMultivariateNormal(model_sparse, init_loc_fn=init_to_median)
svi = SVI(model_sparse, guide_sparse, pyro.optim.Adam({"lr": 0.01}), loss=Trace_ELBO())

for i in range(2000):
    svi.step(data_2d)