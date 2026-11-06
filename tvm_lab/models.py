"""主模型与两个教学 IR：构造计算，不执行编译或计时。"""

import copy

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tvm.script import ir as I, relax as R


def make_model():
    """直接构造固定种子的 ONNX，保留教学所需的算子链与死分支。"""
    rng = np.random.default_rng(0)
    nodes, constants = [], []

    def const(name, value, dtype="float32"):
        constants.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def op(kind, inputs, name, **attrs):
        nodes.append(helper.make_node(kind, inputs, [name], name=name, **attrs))
        return name

    weight = const("weight", rng.normal(0, 0.15, (8, 3, 3, 3)))
    z = op("Conv", ["x", weight], "conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    op("Dropout", [z, const("ratio", 0.5), const("training", False, "bool")], "unused_dropout")
    z = op("Cast", [z], "half", to=TensorProto.FLOAT16)
    z = op("Cast", [z], "roundtrip", to=TensorProto.FLOAT)
    z = op("Reshape", [z, const("flat_shape", [1, 8, 256], "int64")], "flat")
    z = op("Reshape", [z, const("shape", [1, 8, 16, 16], "int64")], "restored")
    b1 = const("bias_a", rng.normal(0, 0.1, (1, 8, 1, 1)))
    b2 = const("bias_b", rng.normal(0, 0.1, (1, 8, 1, 1)))
    bias = op("Add", [b1, b2], "constant_bias_add")
    z = op("Add", [z, bias], "biased")
    # GELU(z) = 0.5 * z * (1 + erf(z / sqrt(2)))。
    scaled = op("Div", [z, const("sqrt_two", np.sqrt(2))], "scaled")
    erf = op("Erf", [scaled], "erf")
    plus_one = op("Add", [erf, const("one", 1)], "plus_one")
    product = op("Mul", [z, plus_one], "product")
    op("Mul", [product, const("half_scalar", .5)], "output")
    graph = helper.make_graph(
        nodes, "conv_cast_gelu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 16, 16])], constants)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model, full_check=True)
    return model


def remove_dead_branch(model):
    """按输出反向追踪依赖；仅用于本例拓扑有序、无子图的纯计算图。"""
    # 当前前端连无用的 Dropout 也拒绝导入。先做 ONNX 可达性清理，原模型保留。
    model = copy.deepcopy(model)
    needed = {value.name for value in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if needed.intersection(node.output):
            kept.append(node)
            needed.update(node.input)
    initializers = [item for item in model.graph.initializer if item.name in needed]
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    onnx.checker.check_model(model, full_check=True)
    return model


# 两个很小的辅助示例：让 pass 的变化和 Cast 舍入能够直接看到。
@I.ir_module
class PassDemo:
    @R.function
    def main(x: R.Tensor((2, 3), "float32")) -> R.Tensor((2, 3), "float32"):
        with R.dataflow():
            bias = R.add(R.const(.125, "float32"), R.const(.25, "float32"))
            unused = R.multiply(x, R.const(7., "float32"))
            result = R.add(x, bias)
            R.output(result)
        return result


@I.ir_module
class CastDemo:
    @R.function
    def main(x: R.Tensor((8,), "float32")) -> R.Tensor((8,), "float32"):
        with R.dataflow():
            half = R.astype(x, "float16")
            result = R.astype(half, "float32")
            R.output(result)
        return result
