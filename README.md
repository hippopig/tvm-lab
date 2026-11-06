# TVM 小实验：Relax 与图优化

构造含 Conv、FP16 往返转换、Reshape、bias、GELU 拆解链及无用 Dropout 分支的 ONNX，观察 Relax → TIR、三个图优化和融合开关，再对小矩阵乘做 auto-tune。所有实验使用 CPU + LLVM。

建议先读 [TVM 入门](TVM入门.md)，再读 [实践报告](TVM.md)，最后结合代码与完整 IR 复现。

## 文件职责

```text
tvm-lab/
├── experiments/       # 实验入口
│   ├── lab.py         # 主模型图优化
│   └── tune.py        # 矩阵乘调优
├── tvm_lab/           # 共用代码
│   ├── common.py      # 环境、配置、校验与计时
│   └── models.py      # ONNX 与教学 IR
├── outputs/           # 模型、IR 与结果
├── TVM入门.md
└── TVM.md
```

两个 Python 目录各有一个 `__init__.py`，以包形式组织。请在项目根目录使用下方的 `python -m` 命令运行。

| 文件 | 内容 |
| --- | --- |
| [TVM入门.md](TVM入门.md) | 独立理论笔记：架构、Relax/TIR、四种图优化、正确性与调优 |
| [TVM.md](TVM.md) | 本次实践：真实 IR 前后对比、测试方法与实测结果 |
| [models.py](tvm_lab/models.py) | ONNX 构图、导入前清理、PassDemo 与 CastDemo |
| [common.py](tvm_lab/common.py) | 单线程环境、CPU target、数值校验和计时 |
| [lab.py](experiments/lab.py) | 主实验、显式 pass 流水线、融合开关和验证 |
| [tune.py](experiments/tune.py) | 128×128 FP32 矩阵乘调优，默认 16 个候选 |
| [requirements.txt](requirements.txt) | 已验证的依赖版本 |

## 运行

使用现有 `cuda-dev` 环境，在项目根目录依次执行：

```bash
conda run --no-capture-output -n cuda-dev python -m experiments.lab
conda run --no-capture-output -n cuda-dev python -m experiments.tune
```

当前环境已安装依赖；若在新环境中复现，先根据 `requirements.txt` 安装。代码使用 TVM `0.25.0.post1` 的 `tvm.tirx` 和 `tvm.s_tir` API。`tvm_lab/common.py` 中 `TARGET` 使用本机支持的 `alderlake`；更换 CPU 时应同步修改。入口必须先导入 `tvm_lab.common`，再导入数值库，以使线程设置生效。

脚本会覆盖对应产物，全部检查通过后写入 `passed: true` 的结果 JSON。调优日志使用临时目录，搜索结束后清理；重跑可能选中不同 schedule，耗时也会波动。实践报告记录本次运行，不会随脚本重跑自动更新。

## 实验产物

| 文件 | 内容 |
| --- | --- |
| [model.onnx](outputs/model.onnx) | 含无用 Dropout 旁支的原始模型 |
| [ir.txt](outputs/ir.txt) | 导入、主模型 fold/dce、完整融合流程和辅助示例 IR |
| [results.json](outputs/results.json) | 主模型误差、计时、实际 pass 列表与 Cast 反例 |
| [tune_ir.txt](outputs/tune_ir.txt) | 调优前后 TIR 与最佳 schedule |
| [tune_results.json](outputs/tune_results.json) | 有效候选数、搜索时间、调优前后误差与耗时 |

主要检查：七组主模型输入、清理前后 ONNX 输出严格一致、辅助 pass 示例、FP16 舍入边界，以及六组矩阵乘输入。默认误差容限为 `atol=1e-5, rtol=1e-4`，严格检查使用零容差。请正常运行脚本，不使用会关闭 `assert` 的 `python -O`。
