# Orin NX TensorRT FP16 转换流程

本流程将现有的三段 TorchScript 模型转换为 ONNX，并且仅在 Orin NX 上构建 FP16 TensorRT engine。不要在 RTX 4060 Ti 上构建并复制 engine 到 Orin NX；TensorRT engine 与 GPU 架构、TensorRT 版本和部分运行时配置相关。

## 0. 先完成板端 F32 基线

在板端运行至少一个模块级 case，保存生成的 JSON：

```bash
conda activate loma
python "ws/scripts/run_baseline_suite.py" \
  --device cuda \
  --case "case1/运动1" \
  --profile-components \
  --warmup 10 \
  --iterations 50
```

这份结果用于证明后续 FP16 质量与性能变化，而不是用桌面 4060 Ti 的耗时替代板端数据。

## 1. 检查导出依赖

ONNX 导出必须在同一个 `loma` Conda 环境中具备 `torch` 和 `onnx`：

```bash
conda activate loma
python -c "import torch, onnx; print(torch.__version__, onnx.__version__)"
```

如果该命令失败，先停止转换，不要混用桌面 CUDA 13 的 TensorRT 动态库。需要在板端为当前 Conda 环境安装与 JetPack/PyTorch 兼容的 `onnx` 包后再继续。

同时确认板端 TensorRT builder：

```bash
which trtexec
trtexec --help | head -1
```

## 2. 导出静态 ONNX

首轮严格固定 `B=1`、`1024 x 1024`、`2048` 个关键点和 `128` 维描述子：

```bash
python "ws/scripts/export_torchscript_onnx.py" \
  --device cuda \
  --input-size 1024 \
  --num-keypoints 2048 \
  --descriptor-dim 128
```

默认输出目录为：

```text
ws/artifacts/onnx/loma_b128_1024_k2048/
```

导出完成后必须检查该目录中的三个 ONNX 文件和 `export_manifest.json`。脚本默认运行 ONNX checker；已有 ONNX 文件时会拒绝覆盖，避免误覆盖可复现实验产物。

## 3. 在 Orin NX 构建 FP16 engine

```bash
bash "ws/scripts/build_tensorrt_fp16.sh" \
  --workspace-mib 1024
```

构建产物、构建日志、layer 信息和 timing cache 均保存到：

```text
ws/engines/loma_b128_1024_k2048/fp16/
```

8GB Orin NX 建议先使用 `1024 MiB` workspace。若构建失败且日志明确提示 tactic/workspace 不足，再逐级尝试 `1536` 或 `2048 MiB`；不要无上限增加 workspace。

## 4. 构建失败时的处理顺序

1. 保留 `logs/loma_<module>_build.log`，不要覆盖旧 engine 或日志。
2. 确认失败模块：detector、descriptor 或 matcher。
3. 检查失败算子是否为 `grid_sample`、`topk`、动态索引或注意力相关操作。
4. 对单一不支持算子，优先考虑 GPU 侧后处理或局部 TensorRT 插件；不要直接退回整个模型到 CPU。
5. 仅在一个模块失败时，继续记录另外两个模块的构建结果，缩小问题范围。

## 5. engine 构建完成后的验证

engine 构建成功只表示 TensorRT 能解析并选择 tactic，不表示输出正确。下一步必须实现 C++ engine runner，对完全相同的图片、letterbox 预处理和输入 shape 执行：

1. TensorRT F32 或 TorchScript F32 参考输出；
2. TensorRT FP16 detector、descriptor、matcher 输出；
3. 关键点坐标、描述子余弦相似度、有效匹配重合率；
4. MAGSAC 内点数、内点率与 SLAM 跟踪质量；
5. P50/P95 延迟、峰值显存、温度和频率。

未经上述数值验证，不将 FP16 engine 接入 SLAM 主循环。
