# flops_profiler/profiler.py

from contextlib import contextmanager
from .hooks import HookManager
from .formatter import print_summary


class FLOPsProfiler:
    """
    forward + backward の FLOPs を層ごとに計測するプロファイラー。

    Usage:
        profiler = FLOPsProfiler(model)

        with profiler.profile():
            loss = criterion(model(x), y)
            loss.backward()

        profiler.summary()
        stats = profiler.stats  # dict[str, LayerStats] で生データも取得可能

        # その他の機能
        profiler.reset()      # summary後にリセットするとhookをすべて解除し、そのまま全て削除する
        profiler.clear_stats()      # hookは残したまま数値だけをリセットする
    """

    def __init__(self, model):
        self.model = model
        self._hook_manager = HookManager()

    @contextmanager
    def profile(self):
        """hookを登録してforward+backwardを計測するcontext manager"""
        self._hook_manager.register(self.model)
        try:
            yield
        finally:
            # hookはcontext脱出後も統計は保持、removeはsummary後に手動で
            pass

    @property
    def stats(self):
        return self._hook_manager.stats

    def summary(self) -> None:
        """計測結果を表示する"""
        if not self._hook_manager.stats:
            print("No stats recorded. Did you use profiler.profile()?")
            return
        print_summary(self._hook_manager.stats)

    def reset(self) -> None:
        """hookを解除してstatsをリセット（次のwith profileで再登録される）"""
        self._hook_manager.remove()

    def clear_stats(self) -> None:
        """hookは維持してstatsだけ0に戻す（エポックをまたぐとき）"""
        self._hook_manager.clear_stats()