"""compute_phi_k의 그룹 격차 추정기 유닛테스트.

수정 내용: 배치별 max-min을 평균내던 방식 → 전 배치에 걸쳐 (y, group) 셀별 활성 합/표본수를
누적한 뒤 마지막에 max-min 을 1회 계산.

옛 방식이 왜 틀렸나: max-min은 비선형이라 E[max-min] > max-min(E[·]) (Jensen 편향).
배치 크기가 64로 고정이라 소표본 셀(UTKFace의 Asian∧old는 배치당 ~3개)에서 그룹평균이
크게 흔들리고, 그 흔들림이 그대로 '격차'로 잡힌다. 결정적으로 이 편향은 배치를 아무리 많이
모아도 사라지지 않는다(각 배치의 셀 크기는 그대로이므로) → 일관성 없는 추정기.

compute_phi_k는 모델/DataLoader/hook이 필요해 통째로 부르기 어려우므로, 실제 코드와 동일한
누적/집계 로직을 여기 재현해 합성 데이터로 검증한다.
(prune.py의 로직이 바뀌면 아래 _accumulate/_pooled_gap 도 함께 갱신할 것)

실행: python tests/test_phi_group_gap.py   또는   python -m pytest tests/test_phi_group_gap.py -q
"""
import torch


# ── prune.compute_phi_k 의 누적/집계 로직과 동일 (수정 후) ────────────────────
def _accumulate(batches, n_groups):
    """batches: [(pooled_act [N,C], target_y [N], sensitive_a [N]), ...]
    반환: cell_sums [n_cells, C], cell_counts [n_cells]   (n_cells = 2*n_groups)"""
    n_cells = 2 * n_groups
    sums = counts = None
    for act, y, a in batches:
        valid = (y >= 0) & (y <= 1) & (a >= 0) & (a < n_groups)
        if not bool(valid.any()):
            continue
        cell_id = (y[valid] * n_groups + a[valid]).long()
        cell_n = torch.bincount(cell_id, minlength=n_cells).to(torch.float64)
        x = act[valid].to(torch.float64)
        if sums is None:
            sums = torch.zeros(n_cells, x.shape[1], dtype=torch.float64)
            counts = torch.zeros(n_cells, dtype=torch.float64)
        sums.index_add_(0, cell_id, x)
        counts += cell_n
    return sums, counts


def _pooled_gap(sums, counts, y_value, n_groups):
    """수정 후: 전 배치 누적 평균으로 max-min 을 한 번만. 유효 그룹 2개 미만이면 None."""
    rows = [
        y_value * n_groups + g
        for g in range(n_groups)
        if float(counts[y_value * n_groups + g]) > 0
    ]
    if len(rows) < 2:
        return None
    means = torch.stack([sums[r] / counts[r] for r in rows], dim=0)
    return means.max(dim=0).values - means.min(dim=0).values


def _per_batch_gap_mean(batches, n_groups, y_value):
    """수정 전(버그): 배치별 max-min 을 계산해 평균. 비교/회귀용."""
    gaps = []
    for act, y, a in batches:
        means = []
        for g in range(n_groups):
            m = (y == y_value) & (a == g)
            if m.any():
                means.append(act[m].mean(dim=0).to(torch.float64))
        if len(means) >= 2:
            s = torch.stack(means, dim=0)
            gaps.append(s.max(dim=0).values - s.min(dim=0).values)
    if not gaps:
        return None
    return torch.stack(gaps, dim=0).mean(dim=0)


# ── 합성 데이터 ─────────────────────────────────────────────────────────────
def _make_batches(group_means, group_probs, n_batches=300, batch_size=64, C=8, seed=0):
    """group_means[g] = 그룹 g의 참 활성 평균 (within-cell std = 1).
    group_probs[g] = 그룹 g 표본 비율. batch_size=64는 compute_phi_k와 동일."""
    gen = torch.Generator().manual_seed(seed)
    probs = torch.tensor(group_probs, dtype=torch.float)
    mus = torch.tensor(group_means, dtype=torch.float)
    batches = []
    for _ in range(n_batches):
        a = torch.multinomial(probs, batch_size, replacement=True, generator=gen)
        y = torch.randint(0, 2, (batch_size,), generator=gen)
        act = torch.randn(batch_size, C, generator=gen) + mus[a][:, None]
        batches.append((act, y, a))
    return batches


# UTKFace race 실제 비율 (White/Black/Asian/Indian). Asian이 가장 희소 → 편향이 가장 큼.
UTK_PROBS = [0.46, 0.21, 0.16, 0.18]
# 실제 UTKFace train(17,610장)은 batch 64 기준 약 275 배치 → 300으로 근사
N_REAL = 300


# ── 테스트 ──────────────────────────────────────────────────────────────────
def test_zero_true_gap_gives_near_zero_gap():
    """핵심: 참 그룹격차 = 0 이면 φ의 격차항도 ≈ 0 이어야 한다."""
    b = _make_batches([0.0] * 4, UTK_PROBS, n_batches=N_REAL, seed=1)
    sums, counts = _accumulate(b, 4)
    for y_value in (0, 1):
        gap = _pooled_gap(sums, counts, y_value, 4)
        assert gap is not None
        # within-cell std=1 기준. 실측 ~0.054 (표본오차) → 0.12는 2배 여유.
        assert gap.mean().item() < 0.12, f"y={y_value} 유령 격차 과대: {gap.mean().item():.4f}"


def test_fixed_estimator_is_consistent_but_old_is_not():
    """수정판은 표본이 늘면 유령 격차가 0으로 수렴(일관성).
    옛 방식은 표본을 늘려도 줄지 않는다(편향) — 이것이 우리가 고친 결함이다."""
    small = _make_batches([0.0] * 4, UTK_PROBS, n_batches=N_REAL, seed=2)
    large = _make_batches([0.0] * 4, UTK_PROBS, n_batches=N_REAL * 10, seed=2)

    new_small = _pooled_gap(*_accumulate(small, 4), 1, 4).mean().item()
    new_large = _pooled_gap(*_accumulate(large, 4), 1, 4).mean().item()
    old_small = _per_batch_gap_mean(small, 4, 1).mean().item()
    old_large = _per_batch_gap_mean(large, 4, 1).mean().item()

    # 수정판: 10배 데이터 → 약 sqrt(10)=3.2배 감소 (최소 2배는 줄어야)
    assert new_large < new_small / 2.0, f"수정판이 수렴하지 않음: {new_small:.4f} → {new_large:.4f}"
    # 옛 방식: 데이터를 늘려도 편향이 그대로 (10% 이상 줄지 않음)
    assert old_large > old_small * 0.9, f"옛 방식 편향이 재현되지 않음: {old_small:.4f} → {old_large:.4f}"
    # 그리고 옛 방식은 수정판보다 압도적으로 크다 (실측 ~0.85 vs ~0.05)
    assert old_small > new_small * 5, f"옛 방식 과대추정이 재현되지 않음: {old_small:.4f} vs {new_small:.4f}"


def test_true_gap_is_recovered_unbiased():
    """참 그룹격차 = 0.30 이면 그 값을 편향 없이 복원해야 한다."""
    true_gap = 0.30
    b = _make_batches([true_gap, 0.0, 0.0, 0.0], UTK_PROBS, n_batches=N_REAL, seed=3)
    gap = _pooled_gap(*_accumulate(b, 4), 1, 4)
    assert abs(gap.mean().item() - true_gap) < 0.05, \
        f"참격차 복원 실패: 추정 {gap.mean().item():.4f} vs 참 {true_gap}"
    # 옛 방식은 노이즈에 묻혀 크게 과대추정
    old = _per_batch_gap_mean(b, 4, 1).mean().item()
    assert old > gap.mean().item() * 2, f"옛 방식 과대추정 미재현: {old:.4f}"


def test_binary_group_case():
    """R=2 (CelebA gender)에서도 참격차 0 → ~0, 있으면 복원."""
    probs = [0.42, 0.58]
    s0, c0 = _accumulate(_make_batches([0.0, 0.0], probs, n_batches=N_REAL, seed=4), 2)
    assert _pooled_gap(s0, c0, 1, 2).mean().item() < 0.12

    s1, c1 = _accumulate(_make_batches([0.25, 0.0], probs, n_batches=N_REAL, seed=5), 2)
    g = _pooled_gap(s1, c1, 1, 2)
    assert abs(g.mean().item() - 0.25) < 0.05, f"R=2 복원 실패: {g.mean().item():.4f}"


def test_missing_group_is_skipped():
    """유효 그룹이 2개 미만이면 None (기존 결측 그룹 규칙 유지)."""
    b = _make_batches([0.0] * 4, [1.0, 0.0, 0.0, 0.0], n_batches=50, seed=6)
    sums, counts = _accumulate(b, 4)
    assert _pooled_gap(sums, counts, 1, 4) is None


if __name__ == "__main__":
    for fn in [
        test_zero_true_gap_gives_near_zero_gap,
        test_fixed_estimator_is_consistent_but_old_is_not,
        test_true_gap_is_recovered_unbiased,
        test_binary_group_case,
        test_missing_group_is_skipped,
    ]:
        fn()
        print(f"PASS  {fn.__name__}")
    print("\n모든 테스트 통과")
