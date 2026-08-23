import numpy as np
import matplotlib.pyplot as plt

data = np.load('Gradient_norm/celeba_gender_grad_diff.npz')
keys = list(data.keys())

diff_keys   = [k for k in keys if k.startswith('abs_diff::')]
female_keys = [k for k in keys if k.startswith('female::')]
male_keys   = [k for k in keys if k.startswith('male::')]

# 블록별로 값 집계
block_diff, block_female, block_male = {}, {}, {}

for prefix, key_list, store in [
    ('abs_diff::', diff_keys,   block_diff),
    ('female::',  female_keys,  block_female),
    ('male::',    male_keys,    block_male),
]:
    for k in key_list:
        name  = k.replace(prefix, '')
        parts = name.split('.')
        block = parts[0] + '.' + parts[1]
        val   = np.abs(data[k].flatten())
        store.setdefault(block, []).extend(val.tolist())

def sort_key(name):
    parts = name.split('.')
    try:    return (parts[0], int(parts[1]))
    except: return (parts[0], 999)

blocks      = sorted(block_diff.keys(), key=sort_key)
mean_diff   = np.array([np.mean(block_diff[b])   for b in blocks])
mean_female = np.array([np.mean(block_female[b]) for b in blocks])
mean_male   = np.array([np.mean(block_male[b])   for b in blocks])

# 레이블 축약
short_labels = [b.replace('features.', 'f').replace('classifier.', 'cls') for b in blocks]

x = np.arange(len(blocks))
w = 0.4

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CelebA — MobileNet-v2 Group Gradient Analysis', fontsize=13)

# --- Fig 1: Female vs Male gradient norm ---
ax = axes[0]
ax.bar(x - w/2, mean_female * 1e4, width=w, label='Female', color='#D85A30', alpha=0.85)
ax.bar(x + w/2, mean_male   * 1e4, width=w, label='Male',   color='#378ADD', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean |gradient| (×10⁻⁴)')
ax.set_title('Fig 1 — Female vs Male gradient norm')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)

# --- Fig 2: 블록별 |Δgradient| ---
ax = axes[1]
colors = ['#D85A30' if 'classifier' in b else '#1D9E75' for b in blocks]
ax.bar(x, mean_diff * 1e4, color=colors, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Mean |Δgradient| (×10⁻⁴)')
ax.set_title('Fig 2 — Unfairness contribution per block')
ax.grid(axis='y', alpha=0.3)

# --- Fig 3: 전체 가중치 |Δ| 분포 ---
ax = axes[2]
all_diff = np.concatenate([data[k].flatten() for k in diff_keys])
p99 = np.percentile(all_diff, 99)  # outlier 제거용 상한
ax.hist(all_diff[all_diff <= p99], bins=50, color='#7F77DD', alpha=0.85, edgecolor='white', linewidth=0.3)
ax.set_yscale('log')
ax.set_xlabel('|Δgradient| (clipped at 99th pct)')
ax.set_ylabel('Weight count (log scale)')
ax.set_title('Fig 3 — Distribution of |Δgradient|')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('gradient_analysis.png', dpi=150, bbox_inches='tight')
plt.show()