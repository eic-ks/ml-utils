# flops_profiler/formatter.py

from .hooks import LayerStats


def _human_readable(n: int) -> str:
    """数値と単位を分けてタプルで返す"""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}", "GFLOPs"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}", "MFLOPs"
    if n >= 1_000:
        return f"{n / 1_000:.2f}", "KFLOPs"
    return f"{n}", "FLOPs"


def _fmt(n: int, num_w: int = 8, unit_w: int = 7) -> str:
    """数値と単位を固定幅で結合する"""
    num, unit = _human_readable(n)
    return f"{num:>{num_w}} {unit:<{unit_w}}"


def print_summary(stats: dict[str, LayerStats]) -> None:
    # 列幅：Layer名 + FLOPs列3つ（数値+単位固定） + ratio
    NAME_W  = 20
    FLOP_W  = 17   # 数値8 + 空白1 + 単位7 = 16、余裕で17
    RATIO_W = 10
    TOTAL_W = NAME_W + FLOP_W * 3 + RATIO_W

    header = (
        f"{'Layer':<{NAME_W}}"
        f"{'Fwd FLOPs':>{FLOP_W}}"
        f"{'Bwd FLOPs':>{FLOP_W}}"
        f"{'Total':>{FLOP_W}}"
        f"{'Bwd/Fwd':>{RATIO_W}}"
    )

    print(header)
    print("-" * TOTAL_W)

    total_fwd = total_bwd = 0
    for name, s in stats.items():
        print(
            f"{name:<{NAME_W}}"
            f"{_fmt(s.fwd_flops):>{FLOP_W}}"
            f"{_fmt(s.bwd_flops):>{FLOP_W}}"
            f"{_fmt(s.total):>{FLOP_W}}"
            f"{s.bwd_fwd_ratio:>{RATIO_W}.2f}x"
        )
        total_fwd += s.fwd_flops
        total_bwd += s.bwd_flops

    total = total_fwd + total_bwd
    print("=" * TOTAL_W)
    print(
        f"{'Total':<{NAME_W}}"
        f"{_fmt(total_fwd):>{FLOP_W}}"
        f"{_fmt(total_bwd):>{FLOP_W}}"
        f"{_fmt(total):>{FLOP_W}}"
        f"{(total_bwd / total_fwd if total_fwd > 0 else 0):>{RATIO_W}.2f}x"
    )