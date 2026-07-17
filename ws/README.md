# LoMa 部署工作区

本目录承载 LoMa 在 Orin NX 上的量化部署、性能测试与部署产物。现阶段先在本地 RTX 4060 Ti 建立 TorchScript F32 基线，再将同一组模型、图像对和测试脚本同步到 Orin NX。

## 目录约定

| 目录 | 用途 |
| --- | --- |
| `models/` | 已导出的 TorchScript 模型，不修改原始模型文件 |
| `scripts/` | 基线、转换、验证与性能测试脚本 |
| `results/` | 自动生成的 JSON 测试结果，不应作为模型输入 |
| `orin_nx_realtime_slam_optimization_plan.md` | Orin NX 实时 SLAM 优化计划 |

## 本地第一步：TorchScript F32 基线

确认已激活 `loma` Conda 环境后，从仓库根目录运行：

```bash
python "ws/scripts/benchmark_torchscript.py" \
  --image-a "assets/a.jpeg" \
  --image-b "assets/b.jpeg" \
  --device cuda \
  --warmup 10 \
  --iterations 50
```

脚本默认加载：

```text
ws/models/loma/b128/loma_detector.pt
ws/models/loma/b128/loma_descriptor.pt
ws/models/loma/b128/loma_matcher.pt
```

输入图像会转换为 RGB、双线性 letterbox 至 `1024 x 1024`、归一化到 `[0, 1]` 的 F32 Tensor。输出 JSON 保存到 `ws/results/torchscript_baseline/`，其中包含：

- detector、descriptor、matcher 和完整单图对 pipeline 的 P50/P95/P99 延迟；
- CUDA 已分配/保留/峰值显存；
- 检测点数、有效匹配数、匹配分数分布；
- MAGSAC 基础矩阵内点数、外点数和内点率。

默认关闭 TF32，使桌面结果可作为 F32 数值参考。若仅需评估 RTX 4060 Ti 的最快 F32 路径，可增加 `--tf32`；该结果不得替代 F32 正确性基线。

## 多场景基线测试

`ws/data/` 已按 `base.jpeg` 与同目录的一张配对图像组织测试样例。以下命令会复用同一组已加载的模型，遍历全部样例并输出汇总 JSON：

```bash
python "ws/scripts/run_baseline_suite.py" \
  --device cuda \
  --warmup 5 \
  --iterations 20
```

首次验证建议先缩小范围，确认模型、CUDA 和数据路径均正常：

```bash
python "ws/scripts/run_baseline_suite.py" \
  --device cuda \
  --warmup 2 \
  --iterations 5 \
  --limit 1
```

完整测试集的每个 case 都会记录完整 pipeline 的 P50/P95/P99、峰值显存、有效匹配数和 MAGSAC 统计。运行器遇到异常时会写入错误记录，并以非零状态码结束。

## 后续顺序

1. 用固定图像对运行并归档本地 F32 基线。
2. 将 `ws/models/`、`ws/scripts/`、测试图像清单和基线 JSON 同步至 Orin NX。
3. 在 Orin NX 上重复同一 TorchScript 基线，确认数值与统计一致。
4. 在 Orin NX 上构建固定 shape 的 FP16 TensorRT engine；不要复用 RTX 4060 Ti 构建的 TensorRT engine。
5. 使用同一测试集执行 TensorRT FP16 正确性、延迟、峰值显存和 SLAM 质量验收。
