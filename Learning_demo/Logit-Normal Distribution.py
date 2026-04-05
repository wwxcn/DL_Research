# logit-normal distribution demo, 用于可视化 Logit-Normal 分布的概率密度函数 (PDF)
# 该分布用于表示在 (0, 1) 区间内的随机变量，其值符合正态分布
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.special import expit  # Sigmoid 函数
from scipy.stats import norm

# --- 1. 定义 Logit-Normal 分布函数 ---

def logit_normal_pdf(x, mu, sigma):
    """
    计算 Logit-Normal 分布的概率密度函数 (PDF)
    公式: f(x) = (1 / (sigma * sqrt(2pi) * x * (1-x))) * exp(-(logit(x) - mu)^2 / (2*sigma^2))
    """
    # 避免 x=0 或 x=1 导致除零错误
    epsilon = 1e-10
    x = np.clip(x, epsilon, 1 - epsilon)
    
    logit_x = np.log(x / (1 - x))
    
    # 正态分布部分
    normal_part = norm.pdf(logit_x, loc=mu, scale=sigma)
    
    # 雅可比行列式部分 (1 / (x * (1-x)))
    jacobian = 1 / (x * (1 - x))
    
    return normal_part * jacobian

def sample_logit_normal(mu, sigma, size=10000):
    """通过变换采样：Normal -> Sigmoid"""
    y = np.random.normal(loc=mu, scale=sigma, size=size)
    return expit(y)

# --- 2. 设置绘图环境 ---

fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(left=0.15, bottom=0.25) # 为滑块留出空间

x = np.linspace(0.001, 0.999, 1000)
mu_init, sigma_init = 0.0, 1.0

# 初始绘制
pdf_line, = ax.plot(x, logit_normal_pdf(x, mu_init, sigma_init), 'b-', lw=2, label='Logit-Normal PDF')
# 添加一个填充区域以便观察
fill = ax.fill_between(x, logit_normal_pdf(x, mu_init, sigma_init), alpha=0.3, color='blue')

# 标题和标签
ax.set_title(f'Logit-Normal Distribution (μ={mu_init:.2f}, σ={sigma_init:.2f})', fontsize=14)
ax.set_xlabel('Value (x) ∈ (0, 1)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(0, 5) # 初始上限，动态调整
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

# 添加统计文本框
text_props = {'fontsize': 10, 'verticalalignment': 'top'}
stats_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, bbox=dict(boxstyle="round", fc="w", ec="gray", alpha=0.8))

# --- 3. 创建滑块控件 ---

# 滑块位置 [left, bottom, width, height]
ax_mu = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_sigma = plt.axes([0.15, 0.10, 0.65, 0.03])

slider_mu = Slider(ax_mu, r'$\mu$ (Location)', -5.0, 5.0, valinit=mu_init, valstep=0.1)
slider_sigma = Slider(ax_sigma, r'$\sigma$ (Scale)', 0.1, 3.0, valinit=sigma_init, valstep=0.05)

# --- 4. 更新逻辑 ---

def update(val):
    mu = slider_mu.val
    sigma = slider_sigma.val
    
    # 重新计算 PDF
    y = logit_normal_pdf(x, mu, sigma)
    
    # 更新线条
    pdf_line.set_ydata(y)
    
    # 更新填充
    fill.remove() # 移除旧的填充
    global fill_obj
    fill_obj = ax.fill_between(x, y, alpha=0.3, color='blue')
    fill = fill_obj # 更新全局引用以便下次移除 (简化处理，实际可优化)
    
    # 动态调整 Y 轴上限，避免曲线被切断
    max_y = np.max(y)
    ax.set_ylim(0, max_y * 1.1 if max_y > 0 else 5)
    
    # 更新标题
    ax.set_title(f'Logit-Normal Distribution (μ={mu:.2f}, σ={sigma:.2f})', fontsize=14)
    
    # 计算近似统计量 (通过数值积分或大量采样估算)
    # 这里用采样估算均值，因为解析解不存在
    samples = sample_logit_normal(mu, sigma, size=50000)
    est_mean = np.mean(samples)
    est_median = expit(mu) # 中位数有解析解: sigmoid(mu)
    
    # 更新统计文本
    shape_desc = ""
    if sigma < 0.5:
        shape_desc = "尖锐单峰 (Sharp Peak)"
    elif sigma > 1.5:
        shape_desc = "双峰或 U 型 (Bimodal / U-shape)"
    else:
        shape_desc = "宽单峰 (Broad Peak)"
        
    stats_str = (
        f"Shape: {shape_desc}\n"
        f"Median (sigmoid(μ)): {est_median:.4f}\n"
        f"Est. Mean (sampled): {est_mean:.4f}\n"
        f"Std Dev (sampled):   {np.std(samples):.4f}"
    )
    stats_text.set_text(stats_str)
    
    fig.canvas.draw_idle()

# 绑定更新事件
slider_mu.on_changed(update)
slider_sigma.on_changed(update)

# 添加重置按钮
reset_ax = plt.axes([0.9, 0.10, 0.05, 0.04])
button_reset = Button(reset_ax, 'Reset', hovercolor='0.975')

def reset(event):
    slider_mu.reset()
    slider_sigma.reset()

button_reset.on_clicked(reset)

plt.show()