"""两个实验共用的环境、正确性校验和计时工具；入口须先于数值库导入本模块。"""

import os
from pathlib import Path

# 在导入数值库之前固定线程数和 CPU，方便比较耗时。
for variable in ("TVM_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

import numpy as np
import tvm

OUT = Path(__file__).resolve().parents[1] / "outputs"
TARGET = tvm.target.Target({"kind": "llvm", "mcpu": "alderlake", "num-cores": 1})


def check(actual, reference, *, strict=False):
    """检查形状、类型及有限数值，返回最大绝对误差；strict 要求逐元素相等。"""
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape, "输出 shape 改变"
    assert actual.dtype == reference.dtype, "输出 dtype 改变"
    assert np.isfinite(actual).all() and np.isfinite(reference).all(), "输出含 NaN/Inf"
    atol, rtol = (0, 0) if strict else (1e-5, 1e-4)
    np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol)
    return float(np.max(np.abs(actual.astype("float64") - reference.astype("float64"))))


def benchmark(runtime, args):
    """预热后测量 main，返回微秒统计；编译和输入转换不在计时范围内。"""
    for _ in range(5):
        runtime["main"](*args)
    samples = runtime.time_evaluator("main", tvm.cpu(), number=1, repeat=10, min_repeat_ms=20)(*args).results
    return {"median_us": float(np.median(samples) * 1e6),
            "p25_us": float(np.quantile(samples, .25) * 1e6),
            "p75_us": float(np.quantile(samples, .75) * 1e6)}
