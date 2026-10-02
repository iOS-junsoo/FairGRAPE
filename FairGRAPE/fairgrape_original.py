"""Run the vendored FairGRAPE algorithm with this repository's experiment data.

The reference is Bernardo1998/FairGRAPE at b677eb9 (``prune_orig_ref.py``).
The dataset loader, label columns and training/evaluation protocol remain local.
"""

import torch


def original_fairgrape_masks(
    model,
    prune_ratio,
    test_csv,
    new_img_dir=None,
    sensitive_classes=2,
    masked_grads=True,
    output_cols_each_task=((0, 2),),
    col_names=('species', 'color'),
    para_batch=1,
    stop_batch=10000,
    delta_p=0,
    scope_layers=None,
):
    """Return reference masks and local logging statistics in module order.

    ``model`` must be the disposable scoring copy with the reference masked
    forwards and masks already installed. The reference changes its training
    state, weights, gradients, BN statistics and trainable masks while scoring;
    these original behaviors are deliberately preserved here. Apply the returned
    masks to the separate model that will be evaluated/retrained.

    ``scope_layers`` is an iterable of resolved layer names; ``None`` means all
    reference-supported layers. A restricted scope (for example MobileNetV2
    blocks) is an explicit adaptation: run the original selection for all layers
    and replace masks outside the scope with their pre-scoring binary masks.
    Capturing those masks beforehand matters because scoring may mutate them.

    Scores, target shares, torch.topk tie handling and greedy selection are
    executed directly by the reference. The reference does not expose final
    selected shares or pick counts; their log fields are left unavailable.
    """
    import prune_orig_ref as reference

    layers = [(name, layer) for name, layer in model.named_modules()
              if type(layer).__name__ in reference.supported_layers]
    layer_names = {name for name, _ in layers}
    scope = layer_names if scope_layers is None else set(scope_layers)
    unknown = scope - layer_names
    if unknown:
        raise ValueError(f"Unknown FairGRAPE scope layers: {sorted(unknown)}")
    if not scope:
        raise ValueError("FairGRAPE pruning scope contains no supported layers")
    if not 0 <= prune_ratio <= 1:
        raise ValueError("prune_ratio must be between 0 and 1")

    before = {}
    for name, layer in layers:
        if not hasattr(layer, 'mask'):
            raise ValueError(f"FairGRAPE scoring layer {name!r} has no mask")
        before[name] = (layer.mask.detach() != 0).to(layer.weight).clone()

    masks = reference.fairness_grad(
        model,
        prune_ratio,
        test_csv,
        new_img_dir=new_img_dir,
        sensitive_classes=sensitive_classes,
        masked_grads=masked_grads,
        output_cols_each_task=output_cols_each_task,
        col_names=col_names,
        para_batch=para_batch,
        impt_type=0,
        stop_batch=stop_batch,
        delta_p=delta_p,
        n_jobs=1,
    )
    if len(masks) != len(layers):
        raise RuntimeError("Reference FairGRAPE returned an unexpected mask count")

    final_masks, stats = [], []
    for (name, layer), mask in zip(layers, masks):
        in_scope = name in scope
        mask = mask.to(layer.weight.device) if in_scope else before[name]
        active_before = int(torch.count_nonzero(before[name]).item())
        kept = int(torch.count_nonzero(mask).item())
        target = reference.LAST_GRAD_TARGET.get(name) if in_scope else None
        final_masks.append(mask)
        stats.append({
            'name': name,
            'numel': layer.weight.numel(),
            'active_before': active_before,
            'keep_target': int(layer.weight.numel() * (1 - prune_ratio)) if in_scope else active_before,
            'kept': kept,
            'removed': active_before - kept,
            'in_scope': in_scope,
            'target_prop': None if target is None else target.copy(),
            'selected_prop': None,
            'picks': '-',
        })
    return final_masks, stats
