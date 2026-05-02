# tests/test_profiler.py
def test_fwd_flops_linear():
    """fc1のforward FLOPsが期待値と一致するか"""
    model = nn.Linear(784, 256)
    profiler = FLOPsProfiler(model)

    with profiler.profile():
        x = torch.randn(32, 784)
        _ = model(x)

    expected = 2 * 32 * 784 * 256   # 手計算した期待値
    assert profiler.stats[""].fwd_flops == expected