def run_sparse_nuts(*, warmup: int = 100, num_samples: int = 300, num_chains: int = 2):
    nuts = NUTS(model_sparse, init_strategy=init_to_median)
    mcmc = MCMC(nuts, num_samples=num_samples, warmup_steps=warmup, num_chains=num_chains)

    t0 = time.time()
    mcmc.run(data_2d)
    dt = time.time() - t0

    print(f"[NUTS sparse] warmup={warmup}, samples={num_samples}, chains={num_chains} | time={dt:.2f}s")

    data_az = az.from_pyro(mcmc)
    print(az.summary(data_az, var_names=["weights"], round_to=3))
    az.plot_trace(data_az, var_names=["weights"])
    return mcmc, dt
