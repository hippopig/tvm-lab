"""主实验：ONNX → Relax → 图优化与 TIR → LLVM → 正确性与计时。"""

# 必须先设置线程和亲和性，再导入 NumPy、ORT 及模型定义。
from tvm_lab.common import OUT, TARGET, benchmark, check

import json

import numpy as np
import onnx
import onnxruntime as ort
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx

from tvm_lab.models import CastDemo, PassDemo, make_model, remove_dead_branch


@tvm.instrument.pass_instrument
class PassLog:
    """记录实际执行的 pass，检查构建阶段没有重新开启融合。"""

    def __init__(self):
        self.names = []

    def run_before_pass(self, mod, info):
        self.names.append(info.name)


def compile_model(mod, *, fold=False, dce=False, fuse=False, log=None):
    """从独立 IR 副本编译，返回 VM、VM lowering 前的 IR 和实际 pass 列表。"""
    mod = tvm.ir.load_json(tvm.ir.save_json(mod))
    recorder = PassLog()
    passes = []
    if fold:
        passes.append(relax.transform.FoldConstant())
    if dce:
        passes.append(relax.transform.DeadCodeElimination())
    passes += [relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern()]
    if fuse:
        passes += [relax.transform.FuseOps(), relax.transform.FuseTIR()]
    with TARGET, tvm.transform.PassContext(opt_level=3, instruments=[recorder]):
        for transform in passes:
            mod = transform(mod)
            # fold/dce 只记录图优化阶段；fusion 记录完整的分组和 TIR 合并过程。
            if log is not None and (fuse or transform.info.name in ("FoldConstant", "DeadCodeElimination")):
                log.write(f"\n# {transform.info.name}\n{mod.script()}\n")
        prepared = mod
        # 这里的通用 VM lowering 不含图融合；不使用含融合的 zero pipeline。
        lowered = relax.pipeline.default_build_pipeline()(mod)
        executable = relax.build(lowered, target=TARGET, relax_pipeline=None)
    if not fuse:
        assert not any(name.endswith(("FuseOps", "FuseTIR")) for name in recorder.names)
    return relax.VirtualMachine(executable, tvm.cpu()), prepared, recorder.names


def ort_session(model):
    """关闭 ORT 图优化，作为独立参考实现；线程设置与 TVM 一致。"""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def main():
    """同一导入 IR 上比较五种配置，全部验证通过后再写入结果 JSON。"""
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").unlink(missing_ok=True)
    model = make_model()
    onnx.save(model, OUT / "model.onnx")
    ready = remove_dead_branch(model)
    mod = from_onnx(ready)
    print("ONNX 节点列表（Dropout 为旁支）：", ", ".join(node.op_type for node in model.graph.node))
    print("\n导入后的 Relax：\n", mod.script())
    original_ort, ready_ort = ort_session(model), ort_session(ready)
    inputs = [np.random.default_rng(i).normal(size=(1, 3, 16, 16)).astype("float32") for i in range(5)]
    inputs += [np.zeros((1, 3, 16, 16), "float32"), np.ones((1, 3, 16, 16), "float32")]
    references = [original_ort.run(None, {"x": x})[0] for x in inputs]
    for x, reference in zip(inputs, references):
        check(ready_ort.run(None, {"x": x})[0], reference, strict=True)

    variants = {"baseline": {}, "fold": {"fold": True}, "dce": {"dce": True},
                "fusion": {"fuse": True}, "combined": {"fold": True, "dce": True, "fuse": True}}
    rows, baseline = [], []
    with (OUT / "ir.txt").open("w", encoding="utf-8") as log:
        log.write(f"# ONNX 导入后的 Relax\n{mod.script()}\n")
        for name, options in variants.items():
            stage_log = log if name in ("fold", "dce", "fusion") else None
            if stage_log is not None:
                log.write(f"\n# 主模型版本：{name}\n")
            vm, prepared, passes = compile_model(mod, **options, log=stage_log)
            outputs = [vm["main"](tvm.runtime.tensor(x)).numpy() for x in inputs]
            errors = [check(y, ref) for y, ref in zip(outputs, references)]
            if name == "baseline":
                baseline = outputs
            changes = [check(y, ref) for y, ref in zip(outputs, baseline)]
            timing = benchmark(vm, [tvm.runtime.tensor(inputs[0])])
            count = sum(isinstance(f, tvm.tirx.PrimFunc) for _, f in prepared.functions_items())
            rows.append({"variant": name, "primfuncs": count, "max_abs_vs_ort": max(errors),
                         "max_abs_vs_baseline": max(changes), **timing, "passes": passes})
            if name == "fusion":
                print("融合后的 Relax：\n", prepared["main"].script())
        demonstrate_passes(log)

    rounding = check_cast()
    result = {"tvm": tvm.__version__, "target": str(TARGET), "seed": 0, "input_cases": len(inputs),
              "atol": 1e-5, "rtol": 1e-4, "variants": rows, "cast_example": rounding, "passed": True}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("\n版本        TIR函数数  max_abs vs ORT  max_abs vs 基线  延迟中位数(us)")
    for row in rows:
        print(f"{row['variant']:10} {row['primfuncs']:8} {row['max_abs_vs_ort']:15.3g} "
              f"{row['max_abs_vs_baseline']:16.3g} {row['median_us']:14.3f}")
    print(f"全部检查通过。详细 IR 和结果保存在 {OUT}")


def demonstrate_passes(log):
    """分别对原始示例执行折叠与 DCE，用结构和数值两方面验证变化。"""
    # bias 在 ONNX 导入时已折叠，死 Dropout 在导入前移除；用小例子观察真正的 pass。
    examples = [("before", PassDemo), ("FoldConstant", relax.transform.FoldConstant()(PassDemo)),
                ("DCE", relax.transform.DeadCodeElimination()(PassDemo))]
    assert examples[1][1].script().count("R.add(") == 1
    assert "R.multiply" not in examples[2][1].script()
    for name, example in examples:
        log.write(f"\n# 独立示例：{name}\n{example.script()}\n")
        vm, _, _ = compile_model(example)
        for i in range(5):
            x = np.random.default_rng(i).normal(size=(2, 3)).astype("float32")
            check(vm["main"](tvm.runtime.tensor(x)).numpy(), x + np.float32(.375), strict=True)
    print("FoldConstant：0.125+0.25 → 0.375；DCE：删除 unused 乘法。")


def check_cast():
    """检查融合前后均保留 FP16 舍入，避免容差掩盖错误的 Cast 删除。"""
    # 包含 FP16 中点与两侧；严格检查才能发现被误删的 fp32→fp16→fp32。
    x = np.array([1.0001, -1.0001, 0, 1.00048828125, 1.0004881620407104,
                  1.0004884004592896, .3333, -.3333], "float32")
    reference = x.astype("float16").astype("float32")
    for fuse in [False, True]:
        vm, _, _ = compile_model(CastDemo, fuse=fuse)
        check(vm["main"](tvm.runtime.tensor(x)).numpy(), reference, strict=True)
    assert x[0] != reference[0], "删除 Cast 的反例应当改变结果"
    print(f"Cast 舍入（融合关/开均保留）：{x[0]:.10f} → {reference[0]:.1f}")
    return {"input": float(x[0]), "roundtrip": float(reference[0])}


if __name__ == "__main__":
    main()
