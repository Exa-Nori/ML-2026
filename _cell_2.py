def model_mf(data, K=3):
    D = data.shape[1]  # Размерность данных
    # Dirichlet(1.0) равномерное распределение весов компонентов смеси
    weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K)))
    with pyro.plate("components", K):
        # Средние для каждого кластера
        locs = pyro.sample("locs", dist.Normal(torch.zeros(D), 1.0).to_event(1))
        # Масштабы для каждого кластера
        scales = pyro.sample("scales", dist.LogNormal(torch.zeros(D), 1.0).to_event(1))

    mixing_dist = dist.Categorical(weights)
    # Объединение компонентов в одну смесь
    component_dist = dist.Normal(locs, scales).to_event(1)
    mixture = dist.MixtureSameFamily(mixing_dist, component_dist)
    
    with pyro.plate("data", len(data)):
        pyro.sample("obs", mixture, obs=data)


pyro.render_model(model_mf, model_args=(data_2d,), render_distributions=True)