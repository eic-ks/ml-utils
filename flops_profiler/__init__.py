# flops_profiler/__init__.py
from .profiler import FLOPsProfiler
from .hooks import LayerStats          # ユーザーが使えるように公開する

__all__ = ["FLOPsProfiler", "LayerStats"]