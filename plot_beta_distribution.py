import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# 设置绘图风格为 Nature 期刊风格
print(plt.style.available)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 21,
    'font.family': 'Times New Roman',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.frameon': False,
    'legend.fontsize': 25,
    'legend.handlelength': 1.5,
    'legend.handletextpad': 0.5,
    'legend.columnspacing': 1.0
})

# 定义 Beta 分布的参数
params = [
    (90.0, 11.0),  # 第一组参数
    (37.0, 21.0),  # 第二组参数
    (2.0, 5.0)
]

# 生成 x 轴数据
x = np.linspace(0, 1, 500)

# 创建图形和轴
fig, ax = plt.subplots(figsize=(6, 4))

# 绘制 Beta 分布曲线
for a, b in params:
    y = beta.pdf(x, a, b)
    # ax.plot(x, y, linewidth=2, label=r'$\alpha$={}, $\beta$={}'.format(a, b))
    ax.plot(x, y, linewidth=2)

# 添加图例
ax.legend(loc='upper left')

# 添加轴标签
ax.set_xlabel('Probability')
ax.set_ylabel('Probability Density')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])  # 隐藏 x 轴刻度
ax.set_yticks([])  # 隐藏 y 轴刻度
# # 添加标题
# ax.set_title('Beta Distribution Curves', fontsize=14, fontweight='bold')

# 显示网格
ax.grid(True, linestyle='--', alpha=0.3)

# 添加带箭头的 x 轴和 y 轴
arrow_length = 0.1  # 箭头长度
arrow_head_width = 0.5  # 箭头宽度
arrow_head_length = 0.5  # 箭头长度

# 添加 x 轴箭头
ax.annotate('', xy=(1.05, 0), xytext=(0, 0),
            arrowprops=dict(facecolor='black', arrowstyle='->',
                            lw=1.5, connectionstyle='arc3'))

# 添加 y 轴箭头
ax.annotate('', xy=(0, 13), xytext=(0, -0.1),
            arrowprops=dict(facecolor='black', arrowstyle='->',
                            lw=1.5, connectionstyle='arc3'))

# 添加 x 轴和 y 轴的标签
ax.text(1.05, 0, 'x', ha='left', va='center')
ax.text(0, 13, 'y', ha='center', va='bottom')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
