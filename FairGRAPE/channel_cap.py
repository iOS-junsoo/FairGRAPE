"""Strict per-iteration channel limits; no tensor or GPU dependencies."""
import math


def validate_layer_remove_ratio(ratio):
    if ratio is None:
        return None
    ratio = float(ratio)
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('layer max remove ratio must be in (0, 1]; omit it to disable')
    return ratio


def channel_removal_limits(original, active, minimum_keep, ratio):
    """Limits use live widths at iteration start, including protected channels.

    No one-channel exception, credit carry, or relaxed second pass is allowed.
    """
    ratio = validate_layer_remove_ratio(ratio)
    if ratio is None:
        return None
    limits = {}
    for block, width in original.items():
        live, floor = active[block], minimum_keep[block]
        if not 0 <= live <= width or not 0 <= floor <= width:
            raise ValueError(f'invalid channel layout for {block}')
        limits[block] = max(0, min(math.floor(ratio * live), live - floor))
    return limits
