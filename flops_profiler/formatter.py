# flops_profiler/formatter.py

from .hooks import LayerStats


def _human_readable(n: int) -> str:
    """FLOPs数を読みやすい単位に変換"""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f} GFLOPs"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} MFLOPs"
    if n >= 1_000:
        return f"{n / 1_000:.2f} KFLOPs"
    return f"{n} FLOPs"


def print_summary(stats: dict[str, LayerStats]) -> None:
    col = [20, 12, 12, 12, 10]
    header = (
        f"{'Layer':<{col[0]}}"
        f"{'Fwd FLOPs':>{col[1]}}"
        f"{'Bwd FLOPs':>{col[2]}}"
        f"{'Total':>{col[3]}}"
        f"{'Bwd/Fwd':>{col[4]}}"
    )
    sep = "-" * sum(col)

    print(header)
    print(sep)

    total_fwd = total_bwd = 0
    for name, s in stats.items():
        print(
            f"{name:<{col[0]}}"
            f"{_human_readable(s.fwd_flops):>{col[1]}}"
            f"{_human_readable(s.bwd_flops):>{col[2]}}"
            f"{_human_readable(s.total):>{col[3]}}"
            f"{s.bwd_fwd_ratio:>{col[4]}.2f}x"
        )
        total_fwd += s.fwd_flops
        total_bwd += s.bwd_flops

    total = total_fwd + total_bwd
    print("=" * sum(col))
    print(
        f"{'Total':<{col[0]}}"
        f"{_human_readable(total_fwd):>{col[1]}}"
        f"{_human_readable(total_bwd):>{col[2]}}"
        f"{_human_readable(total):>{col[3]}}"
        f"{(total_bwd / total_fwd if total_fwd > 0 else 0):>{col[4]}.2f}x"
    )