import matplotlib.pyplot as plt
import numpy as np

# Set ICLR-friendly style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.figsize': (6, 2.8),
    'axes.linewidth': 0.8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Out-of-distribution data
ood_lengths = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
ood_baseline = [0.00687, 0.00235, 0.00584, 0.02010, 0.02749, 0.06196, 0.09048, 0.15779, 0.22786, 0.38029]
ood_guided = [0.00123, 0.00046, 0.00135, 0.00401, 0.00610, 0.01207, 0.01814, 0.04174, 0.06805, 0.14900]

# In-distribution data
id_lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
id_baseline = [0.00242, 0.00433, 0.01009, 0.02213, 0.02758, 0.06196, 0.13047, 0.20035, 0.36422, 0.53700]
id_guided = [0.00049, 0.00094, 0.00208, 0.00629, 0.00826, 0.01585, 0.03015, 0.05698, 0.11760, 0.20092]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))

# Colors
baseline_color = '#d62728'  # red
guided_color = '#1f77b4'    # blue

# Left plot: Out-of-distribution
ax1.plot(ood_lengths, ood_baseline, 'o-', color=baseline_color, label='Unguided', markersize=5, linewidth=1.5)
ax1.plot(ood_lengths, ood_guided, 's-', color=guided_color, label='Guided', markersize=5, linewidth=1.5)
# ax1.fill_between(ood_lengths, ood_guided, ood_baseline, alpha=0.15, color=guided_color)
ax1.set_xlabel('Trajectory Length')
ax1.set_ylabel('MSE-T')
ax1.set_title('(a) Unseen Lengths')
ax1.legend(loc='upper left', framealpha=0.9)
ax1.set_xlim([0, 1000])
ax1.set_ylim([0, 0.58])

# Right plot: In-distribution
ax2.plot(id_lengths, id_baseline, 'o-', color=baseline_color, label='Unguided', markersize=5, linewidth=1.5)
ax2.plot(id_lengths, id_guided, 's-', color=guided_color, label='Guided', markersize=5, linewidth=1.5)
# ax2.fill_between(id_lengths, id_guided, id_baseline, alpha=0.15, color=guided_color)
ax2.set_xlabel('Trajectory Length')
ax2.set_ylabel('MSE-T')
ax2.set_title('(b) Seen Lengths')
ax2.legend(loc='upper left', framealpha=0.9)
ax2.set_xlim([0, 1050])
ax2.set_ylim([0, 0.58])

# import matplotlib.pyplot as plt
# import numpy as np

# # Set ICLR-friendly style
# plt.rcParams.update({
#     'font.family': 'serif',
#     'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
#     'font.size': 10,
#     'axes.labelsize': 11,
#     'axes.titlesize': 11,
#     'xtick.labelsize': 9,
#     'ytick.labelsize': 9,
#     'legend.fontsize': 9,
#     'figure.figsize': (6, 2.8),
#     'axes.linewidth': 0.8,
#     'axes.grid': True,
#     'grid.alpha': 0.3,
#     'grid.linewidth': 0.5,
# })

# # Data from table
# context_fractions = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
# mse_qpos = [0.6619, 0.0406, 0.0400, 0.0565, 0.0565, 0.0565, 0.0565, 0.0565, 0.0565, 0.0565]
# mse_mom = [4.7522, 0.4988, 0.4970, 0.9056, 0.9056, 0.9056, 0.9056, 0.9056, 0.9056, 0.9056]
# mse_total = [5.4142, 0.5394, 0.5370, 0.9622, 0.9622, 0.9622, 0.9622, 0.9622, 0.9622, 0.9622]

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 2.5))

# # Colors
# color = '#1f77b4'  # blue

# # Plot 1: MSE qpos
# ax1.plot(context_fractions, mse_qpos, 'o-', color=color, markersize=5, linewidth=1.5)
# ax1.fill_between(context_fractions, mse_qpos, alpha=0.15, color=color)
# ax1.set_xlabel('Context Fraction')
# ax1.set_ylabel('MSE')
# ax1.set_title('(a) Position')
# ax1.set_xlim([-0.05, 0.95])
# ax1.set_ylim([0, 0.75])

# # Plot 2: MSE mom
# ax2.plot(context_fractions, mse_mom, 'o-', color=color, markersize=5, linewidth=1.5)
# ax2.fill_between(context_fractions, mse_mom, alpha=0.15, color=color)
# ax2.set_xlabel('Context Fraction')
# ax2.set_ylabel('MSE')
# ax2.set_title('(b) Momentum')
# ax2.set_xlim([-0.05, 0.95])
# ax2.set_ylim([0, 5.0])

# # Plot 3: MSE total
# ax3.plot(context_fractions, mse_total, 'o-', color=color, markersize=5, linewidth=1.5)
# ax3.fill_between(context_fractions, mse_total, alpha=0.15, color=color)
# ax3.set_xlabel('Context Fraction')
# ax3.set_ylabel('MSE')
# ax3.set_title('(c) Total')
# ax3.set_xlim([-0.05, 0.95])
# ax3.set_ylim([0, 5.75])

plt.tight_layout()
plt.savefig('/home/gsang/Projects/Perceiver_IO/plots/mse_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
print("Saved to mse_comparison.png")