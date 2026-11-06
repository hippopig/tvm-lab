"""一个小型 CPU auto-tune 示例：128×128 FP32 矩阵乘，默认测量 16 个候选。"""

# 先设置线程与亲和性；调优不依赖主实验或 ONNX 模型。
from tvm_lab.common import OUT, TARGET, benchmark, check

import json
from tempfile import TemporaryDirectory
import time

import numpy as np
import tvm
from tvm import te
from tvm.s_tir import meta_schedule as ms

SIZE = 128
TRIALS = 16


def main():
    """搜索矩阵乘调度，验证候选结果，再独立测量基线与最佳版本。"""
    OUT.mkdir(exist_ok=True)
    (OUT / "tune_results.json").unlink(missing_ok=True)
    # 1. 用 TE 写清数学计算，生成有循环和归约的 PrimFunc。
    a = te.placeholder((SIZE, SIZE), "float32", "A")
    b = te.placeholder((SIZE, SIZE), "float32", "B")
    k = te.reduce_axis((0, SIZE), "k")
    c = te.compute((SIZE, SIZE), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), name="C")
    mod = tvm.IRModule({"main": te.create_prim_func([a, b, c]).with_attr("global_symbol", "main")})
    baseline = tvm.tirx.build(mod, target=TARGET)

    # 2. MetaSchedule 生成候选 schedule，编译并测量，选最快的记录。
    start = time.perf_counter()
    with TemporaryDirectory(prefix="tvm-lab-tuning-") as work_dir:
        database = ms.tir_integration.tune_tir(
            mod, TARGET, work_dir, max_trials_global=TRIALS,
            max_trials_per_task=TRIALS, num_trials_per_iter=TRIALS,
            builder=ms.builder.LocalBuilder(max_workers=1, timeout_sec=60),
            runner=ms.runner.LocalRunner(evaluator_config=ms.runner.EvaluatorConfig(
                number=1, repeat=3, min_repeat_ms=20)),
            cost_model="random", num_tuning_cores=1, seed=0)
        schedule = ms.tir_integration.compile_tir(database, mod, TARGET)
        if schedule is None:
            raise RuntimeError("没有找到有效 schedule")
        records = len(database.get_all_tuning_records())
    search_seconds = time.perf_counter() - start
    tuned = tvm.tirx.build(schedule.mod, target=TARGET)
    (OUT / "tune_ir.txt").write_text(
        f"# 调优前\n{mod.script()}\n\n# 调优后\n{schedule.mod.script()}\n\n# 最佳 schedule\n"
        + "\n".join(schedule.trace.as_python()) + "\n", encoding="utf-8")

    # 3. 调优前后都核对 NumPy，正确后再比较耗时。
    cases = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        cases.append(((rng.normal(size=(SIZE, SIZE)) / 16).astype("float32"),
                      (rng.normal(size=(SIZE, SIZE)) / 16).astype("float32")))
    cases += [(np.eye(SIZE, dtype="float32"), np.ones((SIZE, SIZE), "float32"))]
    # 每组输入与输出缓冲区只分配一次；计时不包含数据转换或分配。
    device_cases = [[tvm.runtime.tensor(x), tvm.runtime.tensor(y),
                     tvm.runtime.empty((SIZE, SIZE), "float32", tvm.cpu())] for x, y in cases]
    rows, baseline_outputs = [], []
    for name, runtime in [("baseline", baseline), ("tuned", tuned)]:
        errors, changes = [], []
        for index, ((x, y), args) in enumerate(zip(cases, device_cases)):
            runtime["main"](*args)
            actual = args[2].numpy()
            errors.append(check(actual, x @ y))
            if name == "baseline":
                baseline_outputs.append(actual)
            changes.append(check(actual, baseline_outputs[index]))
        rows.append({"variant": name, "max_abs_vs_numpy": max(errors),
                     "max_abs_vs_baseline": max(changes), **benchmark(runtime, device_cases[0])})
    result = {"size": SIZE, "trials": TRIALS, "valid_records": records,
              "search_seconds": search_seconds, "variants": rows, "passed": True}
    (OUT / "tune_results.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(f"{row['variant']}: max_abs={row['max_abs_vs_numpy']:.3g}, median={row['median_us']:.3f} us")
    print(f"有效记录 {records}/{TRIALS}，搜索 {search_seconds:.1f}s，"
          f"基线/调优耗时 = {rows[0]['median_us']/rows[1]['median_us']:.2f}x")


if __name__ == "__main__":
    main()
