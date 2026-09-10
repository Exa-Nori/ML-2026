def run_mcmc_with_diagnostics(*, model, data, warmup: int, num_samples: int = 300, num_chains: int = 2, label: str = "MCMC"):
    set_all_seeds(0)

    # NUTS/MCMC чаще быстрее и стабильнее на CPU, и так прогресс отображается лучше
    data_cpu = data.detach().cpu() if isinstance(data, torch.Tensor) else data

    nuts = NUTS(model, init_strategy=init_to_median)
    mcmc = MCMC(
        nuts,
        num_samples=num_samples,
        warmup_steps=warmup,
        num_chains=num_chains,
        disable_progbar=False,
    )

    t0 = time.time()
    mcmc.run(data_cpu)
    dt = time.time() - t0

    print(f"[{label}] warmup={warmup}, samples={num_samples}, chains={num_chains} | time={dt:.2f}s")

    data_az = az.from_pyro(mcmc)
    print(az.summary(data_az, var_names=["weights", "locs"], round_to=2))
    az.plot_trace(data_az, var_names=["weights", "locs"])
    az.plot_autocorr(data_az, var_names=["locs"], combined=True)

    plot_mcmc_results(data_2d.numpy(), mcmc, f"{label} (warmup={warmup}, time={dt:.1f}s)")

    return mcmc, dt

mcmc20, t20 = run_mcmc_with_diagnostics(model=model_full, data=data_2d, warmup=20, label="NUTS Full model")
mcmc100, t100 = run_mcmc_with_diagnostics(model=model_full, data=data_2d, warmup=100, label="NUTS Full model")