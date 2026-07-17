# FP16 TensorRT 转换前置检查

## 当前本地环境状态

已确认的桌面基线环境如下：

| 项目 | 当前状态 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| PyTorch | `2.12.1+cu130` |
| CUDA Runtime | `13.0` |
| TensorRT Python 包 | 未安装 |
| Torch-TensorRT | 未安装 |
| ONNX Python 包 | 未安装 |
| ONNX Runtime | 已安装 |
| 独立 `trtexec` | `/home/shumu/TensorRT-8.6.1.6/bin/trtexec` |

当前 PyTorch CUDA 13 环境与独立 TensorRT 8.6 的运行时来源不同。禁止直接将两者混装或尝试从当前 Conda 环境调用该 `trtexec` 的 Python 库，以免产生 CUDA、cuDNN 或 TensorRT 动态库冲突。

## 转换前必须完成的本地工作

### 1. 固定模块级 F32 基线

先运行一个运动场景和一个遮挡场景的组件剖析：

```bash
python "ws/scripts/run_baseline_suite.py" \
  --device cuda \
  --case "case1/运动1" \
  --profile-components \
  --warmup 10 \
  --iterations 50

python "ws/scripts/run_baseline_suite.py" \
  --device cuda \
  --case "case3/遮挡2" \
  --profile-components \
  --warmup 10 \
  --iterations 50
```

记录 JSON 中的以下字段：

- `module_timing_ms.detector_pair_p50_ms`；
- `module_timing_ms.descriptor_pair_p50_ms`；
- `module_timing_ms.matcher.p50_ms`；
- `full_pipeline.p50_ms` 与 `full_pipeline.p95_ms`；
- 有效匹配数与 RANSAC 内点统计。

这些结果用于确认真实瓶颈并作为后续 FP16 质量回归基线。

### 2. 固定部署输入契约

首个 TensorRT engine 只支持以下固定输入：

| 引擎 | 输入 | 输出 |
| --- | --- | --- |
| detector | `image: [1, 3, 1024, 1024]`，F32/FP16 | `keypoints: [2048, 2]`、`keypoint_probs: [2048]` |
| descriptor | `image: [1, 3, 1024, 1024]`、`keypoints: [2048, 2]` | `descriptors: [2048, 128]` |
| matcher | 两组 `keypoints: [2048, 2]`、`descriptors: [2048, 128]` | `matches: [2048]`、`match_scores: [2048]` |

图像预处理保持与 `ws/scripts/benchmark_torchscript.py` 一致：RGB、双线性 letterbox 到 `1024 x 1024`、归一化到 `[0, 1]`。不在首个 engine 中引入动态分辨率、动态关键点数或 batch。

## 两条转换路径

### 路径 A：Torch-TensorRT

在 CUDA、PyTorch、TensorRT 版本完全匹配的独立环境或官方容器中，直接编译现有 TorchScript 模型。优点是可保留 TorchScript 图并快速做 FP16 可行性验证；缺点是版本耦合强，可能需要为不支持的算子设置 Torch fallback 或插件。

### 路径 B：ONNX + TensorRT

先将三个模块分别导出 ONNX，再使用目标 TensorRT 版本构建 engine。优点是 `trtexec`、layer profile 和 engine 构建过程更透明；缺点是需要额外 ONNX 导出和算子兼容性处理。

无论选择哪条路径，最终 Orin NX engine 必须在 Orin NX 上由与 JetPack 匹配的 TensorRT 构建。RTX 4060 Ti 生成的 engine 仅用于桌面验证，不能复制到 Orin NX 运行。

## 算子兼容性审计

在转换前对每个模块单独生成兼容性报告，重点检查：

| 模块 | 重点算子/风险 |
| --- | --- |
| detector | `max_pool`、`topk`、`gather`、NMS 后处理、子像素采样 |
| descriptor | `grid_sample`、多尺度插值、全图描述子网格 |
| matcher | `scaled_dot_product_attention`、`einsum`、双 softmax、匹配过滤 |

若不支持的算子只属于轻量后处理，优先将其移至 GPU 侧应用逻辑或实现局部插件。不得因为单一不支持算子而将整个 detector、descriptor 或 matcher 回退到 CPU。

## 数值与性能验证顺序

1. Python F32 基线。
2. TorchScript F32 基线。
3. TensorRT F32 engine：定位转换本身造成的差异。
4. TensorRT FP16 engine：定位精度降低与性能收益。
5. Orin NX TensorRT FP16 engine：最终部署验证。

每一层都使用完全相同的图像对和预处理，检查：

| 阶段 | 验证指标 |
| --- | --- |
| detector | 输出 shape、关键点坐标误差、关键点分数分布 |
| descriptor | 输出 shape、描述子余弦相似度 |
| matcher | 有效匹配对重合率、匹配分数分布 |
| 几何验证 | MAGSAC 内点数、内点率与基础矩阵是否可估计 |
| SLAM | 跟踪丢失次数、重定位成功率、轨迹指标 |
| 性能 | P50/P95 延迟、峰值显存、温度与功耗 |

不要求 FP16 与 F32 浮点值逐位一致；若匹配和几何指标超过约定容差，应停止性能优化并定位精度差异。

## 进入转换阶段的条件

满足以下条件后再创建桌面 FP16 engine：

1. 已归档完整 F32 多场景基线和两个代表场景的模块级结果。
2. 已明确所用转换环境的 CUDA、TensorRT、PyTorch 与 Python 版本组合。
3. 已选择路径 A 或路径 B，并确认必要算子的处理方案。
4. 已保留可回退的 TorchScript `.pt`、测试图像清单和结果 JSON。

首轮目标是证明 FP16 engine 的正确性和定位算子障碍，不以桌面 engine 的绝对延迟替代 Orin NX 的性能结论。
