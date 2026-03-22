# Group-Conditioned Activation Gap 기반 공정성 기여도 프록시 계산 지침서
# (레이어별 분석 버전)

---

## 1. 개요

### 1.1 목적

DNN의 **각 레이어, 각 뉴런(채널)**이 공정성 위반(EO 차이)에 얼마나 기여하는지를 정량화하는 프록시 φ를 계산한다. 모든 레이어에 대해 프록시를 구하고, 레이어 간 비교 분석을 수행한다.

### 1.2 핵심 수식

레이어 l의 뉴런 k에 대한 공정성 기여도 프록시:

$$\phi_k^{(l)} = \underbrace{\sum_{y \in \{0,1\}} \left| \mathbb{E}[h_k^{(l)} \mid Y=y,\ A=1] - \mathbb{E}[h_k^{(l)} \mid Y=y,\ A=0] \right|}_{\text{조건부 활성화 차이 (activation gap)}} \times \underbrace{\left| \frac{\partial L}{\partial h_k^{(l)}} \right|}_{\text{출력 영향력 (gradient)}}$$

### 1.3 산출물 목록

| 산출물 | 설명 |
|--------|------|
| 레이어별 gap_Y1 | Y=1 조건에서 그룹 간 활성화 차이 |
| 레이어별 gap_Y0 | Y=0 조건에서 그룹 간 활성화 차이 |
| 레이어별 φ | 공정성 기여도 프록시 |
| 시각화 1 | 레이어별 gap_Y1 분포 |
| 시각화 2 | 레이어별 gap_Y0 분포 |
| 시각화 3 | 레이어별 φ 분포 |

---

## 2. 사전 준비

### 2.1 측정 대상 레이어 정의

모델의 모든 주요 레이어에서 활성화값을 추출한다. Conv 레이어의 경우 채널 단위로 GAP를 적용하여 1차원 벡터로 변환한다.

```python
def get_target_layers(model):
    """
    측정 대상 레이어와 이름을 반환한다.
    모델 구조에 따라 수정 필요.
    
    Returns:
        target_layers: OrderedDict {layer_name: nn.Module}
    """
    from collections import OrderedDict
    target_layers = OrderedDict()
    
    # ── ResNet-18 예시 ──
    target_layers['layer1'] = model.layer1  # [B, 64, 56, 56]
    target_layers['layer2'] = model.layer2  # [B, 128, 28, 28]
    target_layers['layer3'] = model.layer3  # [B, 256, 14, 14]
    target_layers['layer4'] = model.layer4  # [B, 512, 7, 7]
    
    # ── 필요 시 세부 블록 단위 ──
    # target_layers['layer1.0'] = model.layer1[0]
    # target_layers['layer1.1'] = model.layer1[1]
    # ...
    
    return target_layers
```

### 2.2 Hook 기반 활성화값 추출기

```python
class LayerActivationCollector:
    """
    여러 레이어의 활성화값을 동시에 수집하는 hook 기반 수집기.
    Conv 출력은 GAP를 적용하여 [B, C] 형태로 변환한다.
    """
    
    def __init__(self, target_layers):
        """
        Args:
            target_layers: OrderedDict {name: nn.Module}
        """
        self.target_layers = target_layers
        self.activations = {}   # {layer_name: tensor}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        for name, layer in self.target_layers.items():
            hook = layer.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)
    
    def _make_hook(self, name):
        def hook_fn(module, input, output):
            # Conv 출력이면 GAP 적용 → [B, C]
            if output.dim() == 4:
                act = F.adaptive_avg_pool2d(output, 1).squeeze(-1).squeeze(-1)
            elif output.dim() == 2:
                act = output
            else:
                act = output.view(output.size(0), -1)
            self.activations[name] = act.detach().cpu()
        return hook_fn
    
    def clear(self):
        self.activations = {}
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

### 2.3 Hook 기반 그래디언트 추출기

```python
class LayerGradientCollector:
    """
    여러 레이어의 그래디언트를 동시에 수집하는 hook 기반 수집기.
    """
    
    def __init__(self, target_layers):
        self.target_layers = target_layers
        self.gradients = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        for name, layer in self.target_layers.items():
            hook = layer.register_full_backward_hook(self._make_hook(name))
            self.hooks.append(hook)
    
    def _make_hook(self, name):
        def hook_fn(module, grad_input, grad_output):
            grad = grad_output[0]
            # Conv 출력이면 GAP 적용 → [B, C]
            if grad.dim() == 4:
                grad = F.adaptive_avg_pool2d(grad.abs(), 1).squeeze(-1).squeeze(-1)
            elif grad.dim() == 2:
                grad = grad.abs()
            else:
                grad = grad.view(grad.size(0), -1).abs()
            self.gradients[name] = grad.detach().cpu()
        return hook_fn
    
    def clear(self):
        self.gradients = {}
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

---

## 3. 계산 절차

### 전체 흐름

```
데이터
  │
  ▼
[Step 1] 그룹 분리 (Y, A 조합 → 4그룹)
  │
  ▼
[Step 2] 전체 레이어 활성화값 수집 (hook 기반)
  │
  ▼
[Step 3] 레이어별, 그룹별 평균 활성화값 계산
  │
  ▼
[Step 4] 레이어별 조건부 활성화 차이 (gap_Y1, gap_Y0, activation_gap)
  │
  ▼
[Step 5] 전체 레이어 그래디언트 수집 (hook 기반)
  │
  ▼
[Step 6] 레이어별 φ = activation_gap × mean_grad
  │
  ▼
[Step 7] 저장
  │
  ▼
[Step 8] 시각화
         ├── 레이어별 gap_Y1 분포
         ├── 레이어별 gap_Y0 분포
         └── 레이어별 φ 분포
```

---

### Step 1: 데이터 그룹 분리

```python
from collections import defaultdict

group_indices = defaultdict(list)  # {(y, a): [sample_indices]}

for batch_idx, (images, attrs) in enumerate(val_loader):
    y = attrs[:, target_attr_idx]
    a = attrs[:, sensitive_attr_idx]
    
    for i in range(len(y)):
        global_idx = batch_idx * val_loader.batch_size + i
        group_indices[(y[i].item(), a[i].item())].append(global_idx)

# 그룹 샘플 수 확인
for key, indices in sorted(group_indices.items()):
    print(f"  Y={key[0]}, A={key[1]}: {len(indices)} samples")
```

---

### Step 2: 전체 레이어 활성화값 수집

```python
target_layers = get_target_layers(backbone)
collector = LayerActivationCollector(target_layers)

# {layer_name: {(y, a): list of [C] tensors}}
layer_group_activations = {name: defaultdict(list) for name in target_layers}

backbone.eval()
with torch.no_grad():
    for images, attrs in val_loader:
        images = images.to(device)
        y = attrs[:, target_attr_idx]
        a = attrs[:, sensitive_attr_idx]
        
        collector.clear()
        _ = backbone(images)
        
        for layer_name, act in collector.activations.items():
            # act: [B, C]
            for i in range(len(y)):
                key = (y[i].item(), a[i].item())
                layer_group_activations[layer_name][key].append(act[i])

collector.remove_hooks()
```

#### 확인 사항
- [ ] 각 레이어에서 활성화가 정상 수집되는지 (빈 dict가 아닌지)
- [ ] 각 레이어의 채널 수(C) 기록

```python
for layer_name in target_layers:
    sample_key = list(layer_group_activations[layer_name].keys())[0]
    C = layer_group_activations[layer_name][sample_key][0].shape[0]
    print(f"  {layer_name}: {C} channels")
```

---

### Step 3: 레이어별, 그룹별 평균 활성화값 계산

```python
# {layer_name: {(y, a): [C] tensor}}
layer_mean_act = {}
layer_group_counts = {}

for layer_name in target_layers:
    layer_mean_act[layer_name] = {}
    layer_group_counts[layer_name] = {}
    
    for key, act_list in layer_group_activations[layer_name].items():
        stacked = torch.stack(act_list)                          # [N_group, C]
        layer_mean_act[layer_name][key] = stacked.mean(dim=0)    # [C]
        layer_group_counts[layer_name][key] = len(act_list)
```

---

### Step 4: 레이어별 조건부 활성화 차이

```python
# {layer_name: {'gap_Y1': [C], 'gap_Y0': [C], 'activation_gap': [C]}}
layer_gaps = {}

for layer_name in target_layers:
    ma = layer_mean_act[layer_name]
    
    gap_Y1 = (ma[(1, 1)] - ma[(1, 0)]).abs()  # [C]
    gap_Y0 = (ma[(0, 1)] - ma[(0, 0)]).abs()  # [C]
    activation_gap = gap_Y1 + gap_Y0           # [C]
    
    layer_gaps[layer_name] = {
        'gap_Y1': gap_Y1,
        'gap_Y0': gap_Y0,
        'activation_gap': activation_gap,
    }
```

---

### Step 5: 전체 레이어 그래디언트 수집

```python
target_layers = get_target_layers(backbone)
grad_collector = LayerGradientCollector(target_layers)

# {layer_name: list of [B, C] tensors}
layer_grad_list = {name: [] for name in target_layers}

backbone.eval()
classifier.eval()

for images, attrs in val_loader:
    images = images.to(device)
    labels = attrs[:, target_attr_idx].to(device)
    
    grad_collector.clear()
    
    # forward (그래디언트 필요하므로 no_grad 아님)
    features = backbone(images)
    h = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
    logits = classifier(h)
    loss = F.cross_entropy(logits, labels)
    
    # backward
    backbone.zero_grad()
    classifier.zero_grad()
    loss.backward()
    
    for layer_name, grad in grad_collector.gradients.items():
        layer_grad_list[layer_name].append(grad)  # [B, C]

grad_collector.remove_hooks()

# 레이어별 평균 그래디언트
layer_mean_grad = {}
for layer_name in target_layers:
    layer_mean_grad[layer_name] = torch.cat(
        layer_grad_list[layer_name], dim=0
    ).mean(dim=0)  # [C]
```

#### 확인 사항
- [ ] 각 레이어에서 그래디언트가 정상 수집되는지
- [ ] 그래디언트가 전부 0이 아닌지 (특히 앞쪽 레이어에서 vanishing 확인)

```python
for layer_name, mg in layer_mean_grad.items():
    print(f"  {layer_name}: mean_grad mean={mg.mean():.6f}, "
          f"max={mg.max():.6f}, zero_ratio={((mg == 0).sum() / len(mg)):.2%}")
```

---

### Step 6: 레이어별 프록시 계산

```python
# {layer_name: [C] tensor}
layer_fairness_proxy = {}

for layer_name in target_layers:
    phi = layer_gaps[layer_name]['activation_gap'] * layer_mean_grad[layer_name]
    layer_fairness_proxy[layer_name] = phi
```

---

### Step 7: 저장

```python
results = {}

for layer_name in target_layers:
    results[layer_name] = {
        'fairness_proxy': layer_fairness_proxy[layer_name],
        'activation_gap': layer_gaps[layer_name]['activation_gap'],
        'gap_Y1': layer_gaps[layer_name]['gap_Y1'],
        'gap_Y0': layer_gaps[layer_name]['gap_Y0'],
        'mean_grad': layer_mean_grad[layer_name],
        'group_counts': layer_group_counts[layer_name],
        'num_channels': len(layer_fairness_proxy[layer_name]),
    }

torch.save(results, 'layerwise_fairness_proxy_results.pt')

# 기본 통계 출력
print("\n=== 레이어별 프록시 요약 ===")
print(f"{'Layer':<12} {'Channels':>8} {'φ mean':>10} {'φ std':>10} "
      f"{'φ max':>10} {'gap_Y1 mean':>12} {'gap_Y0 mean':>12}")
print("-" * 76)

for layer_name in target_layers:
    r = results[layer_name]
    phi = r['fairness_proxy']
    print(f"{layer_name:<12} {r['num_channels']:>8} "
          f"{phi.mean():>10.6f} {phi.std():>10.6f} {phi.max():>10.6f} "
          f"{r['gap_Y1'].mean():>12.6f} {r['gap_Y0'].mean():>12.6f}")
```

---

### Step 8: 시각화

#### 8.1 레이어별 gap_Y1 분포 (Y=1 조건에서 그룹 간 차이)

```python
import matplotlib.pyplot as plt
import numpy as np

layer_names = list(target_layers.keys())
n_layers = len(layer_names)

fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
if n_layers == 1:
    axes = [axes]

for ax, layer_name in zip(axes, layer_names):
    gap = results[layer_name]['gap_Y1'].numpy()
    
    ax.hist(gap, bins=50, alpha=0.7, color='#2196F3', edgecolor='white')
    ax.axvline(gap.mean(), color='red', linestyle='--', linewidth=1.5, 
               label=f'mean={gap.mean():.4f}')
    ax.set_title(f'{layer_name}\n(C={len(gap)})', fontsize=12)
    ax.set_xlabel('Activation Gap (Y=1)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)

fig.suptitle('Layer-wise Group Activation Gap (Y=1 condition)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('layerwise_gap_Y1_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### 8.2 레이어별 gap_Y0 분포 (Y=0 조건에서 그룹 간 차이)

```python
fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
if n_layers == 1:
    axes = [axes]

for ax, layer_name in zip(axes, layer_names):
    gap = results[layer_name]['gap_Y0'].numpy()
    
    ax.hist(gap, bins=50, alpha=0.7, color='#FF9800', edgecolor='white')
    ax.axvline(gap.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'mean={gap.mean():.4f}')
    ax.set_title(f'{layer_name}\n(C={len(gap)})', fontsize=12)
    ax.set_xlabel('Activation Gap (Y=0)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)

fig.suptitle('Layer-wise Group Activation Gap (Y=0 condition)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('layerwise_gap_Y0_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### 8.3 레이어별 φ 분포

```python
fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4))
if n_layers == 1:
    axes = [axes]

for ax, layer_name in zip(axes, layer_names):
    phi = results[layer_name]['fairness_proxy'].numpy()
    
    ax.hist(phi, bins=50, alpha=0.7, color='#4CAF50', edgecolor='white')
    ax.axvline(phi.mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'mean={phi.mean():.4f}')
    
    # 상위 5% 뉴런 강조
    threshold = np.percentile(phi, 95)
    ax.axvline(threshold, color='orange', linestyle=':', linewidth=1.5,
               label=f'top 5%={threshold:.4f}')
    
    ax.set_title(f'{layer_name}\n(C={len(phi)})', fontsize=12)
    ax.set_xlabel('Fairness Proxy (φ)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)

fig.suptitle('Layer-wise Fairness Proxy (φ) Distribution', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('layerwise_phi_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### 8.4 레이어 간 비교: 요약 boxplot

세 지표(gap_Y1, gap_Y0, φ)를 하나의 figure에 레이어별로 비교한다.

```python
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

metrics = [
    ('gap_Y1', 'Group Gap (Y=1)', '#2196F3'),
    ('gap_Y0', 'Group Gap (Y=0)', '#FF9800'),
    ('fairness_proxy', 'Fairness Proxy (φ)', '#4CAF50'),
]

for ax, (metric_key, title, color) in zip(axes, metrics):
    data = []
    labels = []
    
    for layer_name in layer_names:
        values = results[layer_name][metric_key].numpy()
        data.append(values)
        labels.append(f'{layer_name}\n(C={len(values)})')
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True, 
                    showfliers=False, widths=0.6)
    
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # 평균값 표시
    means = [d.mean() for d in data]
    ax.scatter(range(1, len(means) + 1), means, color='red', 
               zorder=5, s=40, label='mean')
    
    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('layerwise_comparison_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### 8.5 레이어별 gap_Y1 vs gap_Y0 비교 (산점도)

각 레이어에서 뉴런별로 Y=1 조건 차이와 Y=0 조건 차이가 어떤 관계인지 본다.

```python
fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5))
if n_layers == 1:
    axes = [axes]

for ax, layer_name in zip(axes, layer_names):
    g1 = results[layer_name]['gap_Y1'].numpy()
    g0 = results[layer_name]['gap_Y0'].numpy()
    
    ax.scatter(g1, g0, alpha=0.4, s=15, color='#9C27B0')
    
    # 대각선
    max_val = max(g1.max(), g0.max())
    ax.plot([0, max_val], [0, max_val], '--', color='gray', alpha=0.5)
    
    from scipy.stats import spearmanr
    rho, _ = spearmanr(g1, g0)
    ax.set_title(f'{layer_name} (ρ={rho:.3f})', fontsize=12)
    ax.set_xlabel('Gap (Y=1)')
    ax.set_ylabel('Gap (Y=0)')
    ax.grid(True, alpha=0.3)

fig.suptitle('Per-neuron Gap Y=1 vs Gap Y=0 (per layer)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('layerwise_gap_Y1_vs_Y0_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
```

#### 8.6 레이어별 상위 기여 뉴런 히트맵

각 레이어에서 φ 상위 뉴런이 어디에 분포하는지 한눈에 본다.

```python
fig, ax = plt.subplots(figsize=(14, 3))

for i, layer_name in enumerate(layer_names):
    phi = results[layer_name]['fairness_proxy'].numpy()
    C = len(phi)
    
    # 정규화 (레이어 내 상대적 크기)
    if phi.max() > 0:
        phi_norm = phi / phi.max()
    else:
        phi_norm = phi
    
    # 히트맵 한 줄로 표시
    ax.imshow(phi_norm.reshape(1, -1), aspect='auto', cmap='Reds',
              extent=[0, C, i - 0.4, i + 0.4], vmin=0, vmax=1)

ax.set_yticks(range(n_layers))
ax.set_yticklabels(layer_names)
ax.set_xlabel('Channel Index')
ax.set_title('Fairness Proxy (φ) Heatmap per Layer (normalized within layer)')
plt.colorbar(ax.images[0], ax=ax, label='Normalized φ', shrink=0.8)
plt.tight_layout()
plt.savefig('layerwise_phi_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 4. 전체 통합 코드

```python
import torch
import torch.nn.functional as F
from collections import defaultdict, OrderedDict


def compute_layerwise_fairness_proxy(backbone, classifier, val_loader,
                                      target_attr_idx, sensitive_attr_idx,
                                      target_layers, device):
    """
    모든 대상 레이어에 대해 공정성 기여도 프록시를 계산한다.
    
    Args:
        backbone: 특징 추출기
        classifier: 분류 head
        val_loader: DataLoader (images, attrs 반환)
        target_attr_idx: 타겟 속성 인덱스
        sensitive_attr_idx: 민감 속성 인덱스
        target_layers: OrderedDict {name: nn.Module}
        device: torch.device
    
    Returns:
        results: dict {layer_name: {fairness_proxy, activation_gap, gap_Y1, gap_Y0, ...}}
    """
    
    layer_names = list(target_layers.keys())
    
    # ════════════════════════════════════════
    # Part A: 활성화값 수집 (Step 1~4)
    # ════════════════════════════════════════
    
    act_collector = LayerActivationCollector(target_layers)
    layer_group_acts = {name: defaultdict(list) for name in layer_names}
    
    backbone.eval()
    with torch.no_grad():
        for images, attrs in val_loader:
            images = images.to(device)
            y = attrs[:, target_attr_idx]
            a = attrs[:, sensitive_attr_idx]
            
            act_collector.clear()
            _ = backbone(images)
            
            for layer_name, act in act_collector.activations.items():
                for i in range(len(y)):
                    key = (y[i].item(), a[i].item())
                    layer_group_acts[layer_name][key].append(act[i])
    
    act_collector.remove_hooks()
    
    # 그룹별 평균 및 gap 계산
    layer_gaps = {}
    layer_group_counts = {}
    
    for layer_name in layer_names:
        mean_act = {}
        counts = {}
        for key, act_list in layer_group_acts[layer_name].items():
            stacked = torch.stack(act_list)
            mean_act[key] = stacked.mean(dim=0)
            counts[key] = len(act_list)
        
        gap_Y1 = (mean_act[(1, 1)] - mean_act[(1, 0)]).abs()
        gap_Y0 = (mean_act[(0, 1)] - mean_act[(0, 0)]).abs()
        
        layer_gaps[layer_name] = {
            'gap_Y1': gap_Y1,
            'gap_Y0': gap_Y0,
            'activation_gap': gap_Y1 + gap_Y0,
        }
        layer_group_counts[layer_name] = counts
    
    # 메모리 해제
    del layer_group_acts
    
    # ════════════════════════════════════════
    # Part B: 그래디언트 수집 (Step 5)
    # ════════════════════════════════════════
    
    grad_collector = LayerGradientCollector(target_layers)
    layer_grad_list = {name: [] for name in layer_names}
    
    backbone.eval()
    classifier.eval()
    
    for images, attrs in val_loader:
        images = images.to(device)
        labels = attrs[:, target_attr_idx].to(device)
        
        grad_collector.clear()
        backbone.zero_grad()
        classifier.zero_grad()
        
        features = backbone(images)
        h = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        logits = classifier(h)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        
        for layer_name, grad in grad_collector.gradients.items():
            layer_grad_list[layer_name].append(grad)
    
    grad_collector.remove_hooks()
    
    layer_mean_grad = {}
    for layer_name in layer_names:
        layer_mean_grad[layer_name] = torch.cat(
            layer_grad_list[layer_name], dim=0
        ).mean(dim=0)
    
    del layer_grad_list
    
    # ════════════════════════════════════════
    # Part C: 프록시 계산 (Step 6)
    # ════════════════════════════════════════
    
    results = {}
    
    for layer_name in layer_names:
        phi = layer_gaps[layer_name]['activation_gap'] * layer_mean_grad[layer_name]
        
        results[layer_name] = {
            'fairness_proxy': phi,
            'activation_gap': layer_gaps[layer_name]['activation_gap'],
            'gap_Y1': layer_gaps[layer_name]['gap_Y1'],
            'gap_Y0': layer_gaps[layer_name]['gap_Y0'],
            'mean_grad': layer_mean_grad[layer_name],
            'group_counts': layer_group_counts[layer_name],
            'num_channels': len(phi),
        }
    
    return results
```

---

## 5. 실행 예시

```python
# 1. 레이어 정의
target_layers = get_target_layers(backbone)

# 2. 프록시 계산
results = compute_layerwise_fairness_proxy(
    backbone=backbone,
    classifier=classifier,
    val_loader=val_loader,
    target_attr_idx=ATTRACTIVE_IDX,
    sensitive_attr_idx=MALE_IDX,
    target_layers=target_layers,
    device=device,
)

# 3. 저장
torch.save(results, 'layerwise_fairness_proxy_results.pt')

# 4. 요약 출력
print(f"{'Layer':<12} {'Ch':>5} {'φ mean':>10} {'φ max':>10} "
      f"{'gapY1 mean':>11} {'gapY0 mean':>11}")
print("-" * 62)
for name in target_layers:
    r = results[name]
    print(f"{name:<12} {r['num_channels']:>5} "
          f"{r['fairness_proxy'].mean():>10.6f} {r['fairness_proxy'].max():>10.6f} "
          f"{r['gap_Y1'].mean():>11.6f} {r['gap_Y0'].mean():>11.6f}")
```

---

## 6. 체크리스트

### 계산
- [ ] 대상 레이어 목록 정의 완료
- [ ] Hook 등록 및 정상 작동 확인
- [ ] 4개 그룹 모두 충분한 샘플 수 확인
- [ ] 레이어별 활성화값 수집 완료 (NaN/Inf 체크)
- [ ] 레이어별 그래디언트 수집 완료 (전부 0이 아닌지 체크)
- [ ] 레이어별 φ 계산 완료
- [ ] 결과 저장 완료

### 시각화
- [ ] 레이어별 gap_Y1 히스토그램
- [ ] 레이어별 gap_Y0 히스토그램
- [ ] 레이어별 φ 히스토그램
- [ ] 레이어 간 비교 boxplot
- [ ] gap_Y1 vs gap_Y0 산점도
- [ ] φ 히트맵

---

## 7. 주의 사항

1. **Hook 해제**: 실험 종료 후 반드시 `remove_hooks()`를 호출한다. 해제하지 않으면 이후 forward/backward에서 불필요한 연산이 발생한다.

2. **메모리**: 모든 레이어의 활성화값을 동시에 메모리에 올리면 OOM이 발생할 수 있다. 데이터가 크면 레이어를 2~3개씩 나누어 순차 처리하거나, 배치별로 즉시 평균에 누적하는 online 방식을 사용한다.

3. **Conv 레이어의 GAP**: Conv 출력 [B, C, H, W]에 GAP를 적용하면 공간 정보가 사라진다. 채널 단위 프록시만 구하는 것이므로 이는 의도된 동작이다.

4. **Gradient vanishing**: 앞쪽 레이어(예: layer1)에서 그래디언트가 매우 작을 수 있다. 이 경우 φ도 작아지는데, 이는 해당 레이어가 출력에 미치는 직접적 영향이 적다는 정당한 해석이다. 그래디언트가 완전히 0이면 해당 레이어의 프록시는 의미 없으므로 기록만 해두고 분석에서 제외한다.

5. **레이어 간 스케일 차이**: 레이어마다 채널 수와 활성화 스케일이 다르므로, 레이어 간 φ 절대값을 직접 비교하는 것은 주의가 필요하다. 레이어 간 비교 시 레이어 내 정규화(min-max 또는 z-score)를 적용하거나, 상대적 순위로 비교한다.