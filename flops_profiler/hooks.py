# flops_profiler/hooks.py

class LayerStats:
    """1層分のFLOPs統計を保持するデータクラス"""
    def __init__(self):
        self.fwd_flops: int = 0
        self.bwd_flops: int = 0

    @property
    def total(self) -> int:
        return self.fwd_flops + self.bwd_flops

    @property
    def bwd_fwd_ratio(self) -> float:
        return self.bwd_flops / self.fwd_flops if self.fwd_flops > 0 else 0.0


class HookManager:
    """
    Linear層にforward/backward hookを登録し、
    FLOPsをLayerStatsに集計する
    """

    def __init__(self):
        self.stats: dict[str, LayerStats] = {}
        self._handles: list = []

    def register(self, model) -> None:
        """モデル内の全Linear層にhookを登録"""
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Linear":
                self.stats[name] = LayerStats()
                self._handles.append(
                    module.register_forward_hook(self._make_fwd_hook(name))
                )
                self._handles.append(
                    module.register_full_backward_hook(self._make_bwd_hook(name))
                )

    def remove(self) -> None:
        """全hookを解除してstatsをリセット"""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.stats.clear()

    # ------------------------------------------------------------------ #

    def _make_fwd_hook(self, name: str):
        def hook(module, input, output):
            # input[0]: (batch, in_features)
            batch_size  = input[0].shape[0]
            in_feat     = input[0].shape[1]
            out_feat    = output[1]
            # 積和演算: 掛け算+足し算 → ×2
            flops = 2 * batch_size * in_feat * out_feat
            self.stats[name].fwd_flops += flops
        return hook

    def _make_bwd_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            # grad_output[0]: (batch, out_features)
            batch_size  = grad_output[0].shape[0]
            in_feat     = input[0].shape[1]
            out_feat    = output[1]
            # ∂L/∂x と ∂L/∂W の2本分
            flops = 2 * 2 * batch_size * in_feat * out_feat
            self.stats[name].bwd_flops += flops
        return hook