def generate_complex_gmm_data_with_labels(n_samples=1500):
    torch.manual_seed(42)

    # 1) Кластер A: широкий (n=600)
    loc_a = torch.tensor([7.5, 2.5])
    data_a = torch.randn(600, 2) * 1.8 + loc_a
    labels_a = torch.zeros(600)

    # 2) Кластер B: вытянутый по диагонали (n=600)
    loc_b = torch.tensor([5.0, 5.0])
    scale_b = torch.tensor([4.5, 0.6])
    theta = torch.tensor([torch.pi / 4])  # 45 градусов
    rot = torch.tensor([[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()]])
    data_b = (torch.randn(600, 2) * scale_b) @ rot.T + loc_b
    labels_b = torch.ones(600)

    # 3) Кластер C: узкий пик рядом с B (n=300)
    loc_c = torch.tensor([3.5, 5.5])
    data_c = torch.randn(300, 2) * 0.2 + loc_c
    labels_c = torch.ones(300) * 2

    # Объединение
    data = torch.cat([data_a, data_b, data_c])
    labels = torch.cat([labels_a, labels_b, labels_c])
    # Перемешивание с сохранением соответствия меток
    idx = torch.randperm(n_samples)
    return data[idx], labels[idx]


data_2d, true_labels = generate_complex_gmm_data_with_labels()

plt.figure(figsize=(10, 7))
colors = ['#1f77b4', '#ff7f0e', '#d62728']
for i, lbl in enumerate(['A', 'B', 'C']):
    mask = (true_labels == i)
    plt.scatter(data_2d[mask, 0], data_2d[mask, 1], 
                s=10, alpha=0.6, c=colors[i], label=f'Cluster {lbl}')


plt.title("Ground Truth")
plt.legend()
#plt.axis('equal')
plt.xlim(-8, 15)
plt.grid(True, alpha=0.2)
plt.show()