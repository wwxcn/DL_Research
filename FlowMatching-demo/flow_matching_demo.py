"""
Flow Matching (Rectified Flow) 使用 DiT (Diffusion Transformer) 实现
基于论文: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from dit_model import DiT
from flow_matching import FlowMatching
import os


SAVE_FIGURES = False
MODEL_DIR = 'saved_models'
MODEL_PATH = os.path.join(MODEL_DIR, 'swiss_roll_dit.pt')


def create_2d_dataset(n_samples=20000, dataset_type='swiss_roll'):
    """
    创建2D测试数据集
    """
    if dataset_type == 'swiss_roll':
        # Swiss Roll 数据集
        t = np.random.rand(n_samples) * 4 * np.pi
        x = t * np.cos(t)
        y = t * np.sin(t)
        data = np.stack([x, y], axis=1)
        # 归一化到 [-3, 3] 范围
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        data = data * 2.0

    elif dataset_type == 'moons':
        # 双月数据集
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=n_samples, noise=0.05)
        # 归一化
        data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
        data = data * 2.0

    elif dataset_type == 'gaussian_mixture':
        # 高斯混合
        n_per_cluster = n_samples // 4
        data = []
        for i in range(4):
            angle = i * np.pi / 2
            center = np.array([np.cos(angle), np.sin(angle)]) * 2.5
            cluster = np.random.randn(n_per_cluster, 2) * 0.5 + center
            data.append(cluster)
        data = np.vstack(data)

    return torch.tensor(data, dtype=torch.float32)


def plot_original_vs_generated(original_data, generated_data, title="Original vs Generated"):
    """
    绘制原始数据与生成数据的散点图对比
    original_data: [N, 2] 原始数据
    generated_data: [N, 2] 生成数据
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(original_data[:, 0], original_data[:, 1], alpha=0.5, s=10, c='blue')
    axes[0].set_title('Original Data', fontsize=14)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(generated_data[:, 0], generated_data[:, 1], alpha=0.5, s=10, c='orange')
    axes[1].set_title('Generated Data', fontsize=14)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=150)
        print(f"Figure saved: {title.replace(' ', '_').lower()}.png")
    plt.show()


def plot_interpolation_process(model, device, n_samples=500, n_steps=5):
    """
    绘制 Flow Matching 插值过程可视化，理论结果！
    展示从噪声 x0 到数据 x1 的中间过程
    """
    model.eval()

    with torch.no_grad():
        x0 = torch.randn(n_samples, 2, device=device)
        t = np.random.rand(n_samples) * 4 * np.pi
        x1_np = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
        x1_np = (x1_np - x1_np.mean(axis=0)) / (x1_np.std(axis=0) + 1e-8)
        x1 = torch.tensor(x1_np * 2.0, dtype=torch.float32, device=device)

        t_values = np.linspace(0, 1, n_steps)
        interpolates = []

        for t_val in t_values:
            t = torch.ones(n_samples, device=device) * t_val
            x_t = (1 - t.view(-1, 1)) * x0 + t.view(-1, 1) * x1
            interpolates.append(x_t.cpu().numpy())

    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))

    for i, (ax, x_t) in enumerate(zip(axes, interpolates)):
        ax.scatter(x_t[:, 0], x_t[:, 1], alpha=0.5, s=10, c='green')
        ax.set_title(f't = {t_values[i]:.2f}', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    fig.suptitle('Flow Matching Interpolation: x₀ (noise) → x₁ (data)', fontsize=14)
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig('interpolation_process.png', dpi=150)
        print("Figure saved: interpolation_process.png")
    plt.show()

    model.train()


def plot_sample_trajectory(model, device, n_trajectories=5, num_steps=50):
    """
    采样轨迹可视化：展示多个样本从噪声到数据的完整路径
    """
    model.eval()

    with torch.no_grad():
        x = torch.randn(n_trajectories, 2, device=device)
        dt = 1.0 / num_steps
        trajectory = [x.cpu().numpy()]

        for i in range(num_steps):
            t = torch.ones(n_trajectories, device=device) * (i * dt)
            v = model(x, t)
            x = x + dt * v
            trajectory.append(x.cpu().numpy())

    model.train()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, n_trajectories))

    for i in range(n_trajectories):
        traj = np.array([step[i] for step in trajectory])
        ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], alpha=0.7, linewidth=2)
        ax.scatter(traj[0, 0], traj[0, 1], color=colors[i], marker='o', s=50, label=f'Traj {i+1} (start)')
        ax.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker='*', s=100, edgecolor='black')

    ax.set_title('Sample Trajectories: Noise → Data', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig('sample_trajectories.png', dpi=150)
        print("Figure saved: sample_trajectories.png")
    plt.show()


def plot_vector_field(model, device, grid_size=20, t_value=0.5):
    """
    2D流场可视化：展示模型学习到的向量场
    """
    model.eval()

    x = np.linspace(-4, 4, grid_size)
    y = np.linspace(-4, 4, grid_size)
    xx, yy = np.meshgrid(x, y)

    grid_points = torch.tensor(np.stack([xx.flatten(), yy.flatten()], axis=1),
                                dtype=torch.float32, device=device)
    t = torch.ones(grid_points.shape[0], device=device) * t_value

    with torch.no_grad():
        v = model(grid_points, t).cpu().numpy()

    vx = v[:, 0].reshape(grid_size, grid_size)
    vy = v[:, 1].reshape(grid_size, grid_size)

    model.train()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(xx, yy, vx, vy, color='blue', alpha=0.6, scale=50, width=0.002)
    ax.streamplot(xx, yy, vx, vy, color='red', linewidth=1, density=1.5, arrowstyle='->')
    ax.set_title(f'Learned Vector Field at t={t_value:.2f}', fontsize=14)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE_FIGURES:
        plt.savefig(f'vector_field_t{t_value:.2f}.png', dpi=150)
        print(f"Figure saved: vector_field_t{t_value:.2f}.png")
    plt.show()


def test_multiple_datasets(device, dataset_types=['moons', 'gaussian_mixture']):
    """
    在多个数据集上测试 Flow Matching 模型，重新训练模型，检测泛化能力
    """
    print("\n" + "="*60)
    print("Testing with Different Datasets")
    print("="*60 + "\n")

    for dataset_type in dataset_types:
        print(f"\nTesting with {dataset_type} dataset...")
        test_data = create_2d_dataset(n_samples=20000, dataset_type=dataset_type)
        test_dataset = TensorDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)

        test_model = DiT(
            input_dim=2,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0
        )
        test_flow = FlowMatching(test_model, device=device)

        print(f"Training on {dataset_type}...")
        test_losses = test_flow.train(test_loader, num_epochs=100, lr=1e-3)

        test_generated = test_flow.sample(500, num_steps=100).cpu().numpy()
        print(f"Final loss: {test_losses[-1]:.6f}")
        print(f"Loss reduction: {100*(test_losses[0]-test_losses[-1])/test_losses[0]:.2f}%")
        print(f"Generated samples: {len(test_generated)}")

        print(f"Generating visualization for {dataset_type}...")
        plot_original_vs_generated(
            test_data.cpu().numpy()[:500],
            test_generated,
            title=f"{dataset_type.replace('_', ' ').title()}: Original vs Generated"
        )

    print("\n" + "="*60)
    print("Multi-dataset Testing Completed!")
    print("="*60)


def save_model(model, path):
    """保存模型到指定路径"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to: {path}")


def load_model(model, path, device):
    """从指定路径加载模型"""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from: {path}")
    return model


def main():
    """
    主函数: 演示使用 DiT 的 Flow Matching 训练和采样
    """
    # 设置计算设备 (GPU/CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\n" + "="*60)
    print("Flow Matching with Diffusion Transformer (DiT)")
    print("="*60)

    # 创建 Swiss Roll 数据集
    print("\nCreating dataset...")
    data = create_2d_dataset(n_samples=20000, dataset_type='swiss_roll')
    print(f"Dataset created with shape: {data.shape}")

    # 创建数据加载器
    dataset = TensorDataset(data)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    # 创建 DiT 模型
    print("\nInitializing DiT model...")
    model = DiT(
        input_dim=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 创建 Flow Matching 实例
    flow_matching = FlowMatching(model, device=device)

    # 检查是否已有保存的模型，如有则加载，否则训练新模型
    if os.path.exists(MODEL_PATH):
        print(f"\nFound saved model at: {MODEL_PATH}")
        print("Loading model instead of training...")
        flow_matching.model = load_model(model, MODEL_PATH, device)
    else:
        # 训练模型
        print("\n" + "="*60)
        print("Training Flow Matching Model")
        print("="*60 + "\n")

        losses = flow_matching.train(data_loader, num_epochs=200, lr=1e-3)
        print(f"\nTraining completed. Final loss: {losses[-1]:.6f}")
        print(f"Loss reduction: {100*(losses[0]-losses[-1])/losses[0]:.2f}%")

        # 保存训练好的模型
        print("\nSaving model...")
        save_model(flow_matching.model, MODEL_PATH)

    # 使用训练好的模型生成样本
    print("\n" + "="*60)
    print("Generating Samples")
    print("="*60 + "\n")

    generated = flow_matching.sample(1000, num_steps=100).cpu().numpy()
    print(f"Generated {len(generated)} samples")

    # 打印原始数据与生成数据的统计信息对比
    print(f"\nOriginal data - Mean: {data.mean().numpy():.2f}, Std: {data.std().numpy():.2f}")
    print(f"Generated data - Mean: {generated.mean():.2f}, Std: {generated.std():.2f}")

    # 生成可视化结果
    print("\n" + "="*60)
    print("Visualization")
    print("="*60 + "\n")

    # 可视化1: 原始数据 vs 生成数据散点图对比
    print("Generating visualization: Original vs Generated (Swiss Roll)...")
    plot_original_vs_generated(
        data.cpu().numpy()[:1000],
        generated,
        title="Swiss Roll: Original vs Generated"
    )

    # 可视化2: Flow Matching 理论插值过程 (不使用模型)
    print("\nGenerating visualization: Interpolation Process...")
    plot_interpolation_process(model, device, n_samples=500, n_steps=5)

    # 可视化3: 采样轨迹可视化 (使用模型推理)
    print("\nGenerating visualization: Sample Trajectories...")
    plot_sample_trajectory(model, device, n_trajectories=50, num_steps=50)

    # 可视化4: 不同时间步的向量场可视化
    print("\nGenerating visualization: Vector Field at t=0.25...")
    plot_vector_field(model, device, grid_size=20, t_value=0.25)

    # 可视化4b: t=0.5 时刻的向量场
    print("\nGenerating visualization: Vector Field at t=0.5...")
    plot_vector_field(model, device, grid_size=20, t_value=0.5)

    # 可视化4c: t=0.75 时刻的向量场
    print("\nGenerating visualization: Vector Field at t=0.75...")
    plot_vector_field(model, device, grid_size=20, t_value=0.75)

    # 在其他数据集 (moons, gaussian_mixture) 上测试模型泛化能力
    test_multiple_datasets(device)

    print("\n" + "="*60)
    print("All Tests Completed Successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
