from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar

import cv2
import numpy as np
import torch
from PIL import Image


WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = WORKSPACE_DIR / "models" / "loma" / "b128"
DEFAULT_RESULTS_DIR = WORKSPACE_DIR / "results" / "torchscript_baseline"
Result = TypeVar("Result")


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    input_size: int
    scale: float
    pad_left: int
    pad_top: int


@dataclass
class PipelineOutput:
    keypoints_a: torch.Tensor
    keypoints_b: torch.Tensor
    keypoint_scores_a: torch.Tensor
    keypoint_scores_b: torch.Tensor
    descriptors_a: torch.Tensor
    descriptors_b: torch.Tensor
    matches_a: torch.Tensor
    match_scores_a: torch.Tensor


class TorchScriptLoMa:
    def __init__(self, models_dir: Path, device: torch.device) -> None:
        self.detector = self._load(models_dir / "loma_detector.pt", device)
        self.descriptor = self._load(models_dir / "loma_descriptor.pt", device)
        self.matcher = self._load(models_dir / "loma_matcher.pt", device)

    @staticmethod
    def _load(path: Path, device: torch.device) -> torch.jit.ScriptModule:
        if not path.is_file():
            raise FileNotFoundError(f"找不到模型文件: {path}")
        return torch.jit.load(str(path), map_location=device).eval()

    def detect(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.detector(image)

    def describe(
        self, image: torch.Tensor, keypoints: torch.Tensor
    ) -> torch.Tensor:
        return self.descriptor(image, keypoints)

    def match(
        self,
        keypoints_a: torch.Tensor,
        descriptors_a: torch.Tensor,
        keypoints_b: torch.Tensor,
        descriptors_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.matcher(keypoints_a, descriptors_a, keypoints_b, descriptors_b)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对导出的 LoMa TorchScript 三段模型进行单图对基线测试。"
    )
    parser.add_argument("--image-a", type=Path, required=True, help="第一张输入图像")
    parser.add_argument("--image-b", type=Path, required=True, help="第二张输入图像")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="包含 loma_detector.pt、loma_descriptor.pt、loma_matcher.pt 的目录",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="JSON 测试结果输出目录",
    )
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="默认优先使用 CUDA",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="允许 CUDA 在 F32 算子中使用 TF32；默认关闭以保留 F32 参考基线",
    )
    parser.add_argument("--ransac-threshold", type=float, default=0.5)
    parser.add_argument("--ransac-confidence", type=float, default=0.999999)
    parser.add_argument("--ransac-max-iters", type=int, default=10000)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 环境未检测到可用 CUDA 设备。")
    return torch.device(value)


def load_letterboxed_image(
    path: Path, input_size: int, device: torch.device
) -> tuple[torch.Tensor, LetterboxTransform]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到输入图像: {path}")
    if input_size <= 0:
        raise ValueError("input_size 必须为正整数。")

    with Image.open(path) as source_image:
        source_image = source_image.convert("RGB")
        source_width, source_height = source_image.size
        scale = min(input_size / source_width, input_size / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        resized = source_image.resize(
            (resized_width, resized_height), Image.Resampling.BILINEAR
        )

    pad_left = (input_size - resized_width) // 2
    pad_top = (input_size - resized_height) // 2
    canvas = Image.new("RGB", (input_size, input_size))
    canvas.paste(resized, (pad_left, pad_top))
    pixels = np.asarray(canvas, dtype=np.float32) / 255.0
    image = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
    transform = LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        input_size=input_size,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return image, transform


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_inference(operation: Callable[[], Result]) -> Result:
    with torch.inference_mode():
        return operation()


def summarize_latency(samples_ms: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean_ms": round(float(values.mean()), 3),
        "p50_ms": round(float(np.percentile(values, 50)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "p99_ms": round(float(np.percentile(values, 99)), 3),
        "min_ms": round(float(values.min()), 3),
        "max_ms": round(float(values.max()), 3),
    }


def benchmark(
    operation: Callable[[], Result],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[Result, dict[str, float | int]]:
    if warmup < 0:
        raise ValueError("warmup 不能为负数。")
    if iterations <= 0:
        raise ValueError("iterations 必须大于零。")

    for _ in range(warmup):
        run_inference(operation)
    synchronize(device)

    samples_ms: list[float] = []
    result: Result | None = None
    for _ in range(iterations):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = run_inference(operation)
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            result = run_inference(operation)
            samples_ms.append((time.perf_counter() - start_time) * 1000)

    if result is None:
        raise RuntimeError("未获得任何推理结果。")
    return result, summarize_latency(samples_ms)


def capture_peak_memory(
    operation: Callable[[], Result], device: torch.device
) -> tuple[Result, dict[str, float]]:
    if device.type != "cuda":
        return run_inference(operation), {}

    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = run_inference(operation)
    synchronize(device)
    bytes_per_mib = 1024 * 1024
    return result, {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / bytes_per_mib, 2),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / bytes_per_mib, 2),
        "peak_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / bytes_per_mib, 2
        ),
        "peak_reserved_mib": round(
            torch.cuda.max_memory_reserved(device) / bytes_per_mib, 2
        ),
    }


def to_source_coordinates(
    keypoints: torch.Tensor, transform: LetterboxTransform
) -> np.ndarray:
    points = keypoints.detach().float().cpu().numpy()
    pixels = (points + 1.0) * (transform.input_size / 2.0)
    pixels[:, 0] = (pixels[:, 0] - transform.pad_left) / transform.scale
    pixels[:, 1] = (pixels[:, 1] - transform.pad_top) / transform.scale
    return pixels


def match_statistics(
    output: PipelineOutput,
    transform_a: LetterboxTransform,
    transform_b: LetterboxTransform,
    threshold: float,
    confidence: float,
    max_iters: int,
) -> dict[str, object]:
    matches_a = output.matches_a.detach().cpu().numpy()
    match_scores_a = output.match_scores_a.detach().float().cpu().numpy()
    source_a = to_source_coordinates(output.keypoints_a, transform_a)
    source_b = to_source_coordinates(output.keypoints_b, transform_b)

    valid = matches_a >= 0
    indices_a = np.flatnonzero(valid)
    indices_b = matches_a[valid]
    matched_a = source_a[indices_a]
    matched_b = source_b[indices_b]
    matched_scores = match_scores_a[valid]

    inside_a = (
        (matched_a[:, 0] >= 0)
        & (matched_a[:, 0] < transform_a.source_width)
        & (matched_a[:, 1] >= 0)
        & (matched_a[:, 1] < transform_a.source_height)
    )
    inside_b = (
        (matched_b[:, 0] >= 0)
        & (matched_b[:, 0] < transform_b.source_width)
        & (matched_b[:, 1] >= 0)
        & (matched_b[:, 1] < transform_b.source_height)
    )
    inside = inside_a & inside_b
    matched_a = matched_a[inside]
    matched_b = matched_b[inside]
    matched_scores = matched_scores[inside]

    ransac: dict[str, object] = {
        "threshold_px": threshold,
        "confidence": confidence,
        "max_iters": max_iters,
        "inliers": 0,
        "outliers": int(len(matched_a)),
        "inlier_ratio": 0.0,
        "fundamental_matrix": None,
    }
    if len(matched_a) >= 8:
        matrix, inlier_mask = cv2.findFundamentalMat(
            matched_a.astype(np.float32),
            matched_b.astype(np.float32),
            cv2.USAC_MAGSAC,
            threshold,
            confidence,
            max_iters,
        )
        if inlier_mask is not None:
            inliers = int(inlier_mask.sum())
            ransac.update(
                {
                    "inliers": inliers,
                    "outliers": int(len(matched_a) - inliers),
                    "inlier_ratio": round(inliers / len(matched_a), 6),
                    "fundamental_matrix": matrix.tolist() if matrix is not None else None,
                }
            )

    return {
        "detected_keypoints_a": int(len(output.keypoints_a)),
        "detected_keypoints_b": int(len(output.keypoints_b)),
        "valid_matches_before_padding_filter": int(valid.sum()),
        "valid_matches": int(len(matched_a)),
        "match_score": {
            "mean": round(float(matched_scores.mean()), 6)
            if len(matched_scores)
            else None,
            "min": round(float(matched_scores.min()), 6)
            if len(matched_scores)
            else None,
            "max": round(float(matched_scores.max()), 6)
            if len(matched_scores)
            else None,
        },
        "ransac": ransac,
    }


def make_pipeline_operation(
    model: TorchScriptLoMa, image_a: torch.Tensor, image_b: torch.Tensor
) -> Callable[[], PipelineOutput]:
    def operation() -> PipelineOutput:
        keypoints_a, keypoint_scores_a = model.detect(image_a)
        keypoints_b, keypoint_scores_b = model.detect(image_b)
        descriptors_a = model.describe(image_a, keypoints_a)
        descriptors_b = model.describe(image_b, keypoints_b)
        matches_a, match_scores_a = model.match(
            keypoints_a, descriptors_a, keypoints_b, descriptors_b
        )
        return PipelineOutput(
            keypoints_a=keypoints_a,
            keypoints_b=keypoints_b,
            keypoint_scores_a=keypoint_scores_a,
            keypoint_scores_b=keypoint_scores_b,
            descriptors_a=descriptors_a,
            descriptors_b=descriptors_b,
            matches_a=matches_a,
            match_scores_a=match_scores_a,
        )

    return operation


def write_result(results_dir: Path, result: dict[str, object]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"torchscript_baseline_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)
    return output_path


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    model = TorchScriptLoMa(args.models_dir, device)
    image_a, transform_a = load_letterboxed_image(
        args.image_a, args.input_size, device
    )
    image_b, transform_b = load_letterboxed_image(
        args.image_b, args.input_size, device
    )
    pipeline_operation = make_pipeline_operation(model, image_a, image_b)

    detector_a_output, detector_a_timing = benchmark(
        lambda: model.detect(image_a), args.warmup, args.iterations, device
    )
    detector_b_output, detector_b_timing = benchmark(
        lambda: model.detect(image_b), args.warmup, args.iterations, device
    )
    keypoints_a, _ = detector_a_output
    keypoints_b, _ = detector_b_output
    descriptors_a, descriptor_a_timing = benchmark(
        lambda: model.describe(image_a, keypoints_a),
        args.warmup,
        args.iterations,
        device,
    )
    descriptors_b, descriptor_b_timing = benchmark(
        lambda: model.describe(image_b, keypoints_b),
        args.warmup,
        args.iterations,
        device,
    )
    _, matcher_timing = benchmark(
        lambda: model.match(keypoints_a, descriptors_a, keypoints_b, descriptors_b),
        args.warmup,
        args.iterations,
        device,
    )
    _, pipeline_timing = benchmark(
        pipeline_operation, args.warmup, args.iterations, device
    )
    output, memory = capture_peak_memory(pipeline_operation, device)
    statistics = match_statistics(
        output,
        transform_a,
        transform_b,
        args.ransac_threshold,
        args.ransac_confidence,
        args.ransac_max_iters,
    )

    result: dict[str, object] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runtime": {
            "torch_version": torch.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "tf32": args.tf32 if device.type == "cuda" else None,
        },
        "models_dir": str(args.models_dir.resolve()),
        "inputs": {
            "image_a": str(args.image_a.resolve()),
            "image_b": str(args.image_b.resolve()),
            "preprocess": "RGB, bilinear letterbox, [0, 1] float32",
            "transform_a": asdict(transform_a),
            "transform_b": asdict(transform_b),
        },
        "timing_ms": {
            "detector_a": detector_a_timing,
            "detector_b": detector_b_timing,
            "descriptor_a": descriptor_a_timing,
            "descriptor_b": descriptor_b_timing,
            "matcher": matcher_timing,
            "full_pipeline": pipeline_timing,
        },
        "memory_mib": memory,
        "matching": statistics,
    }
    output_path = write_result(args.results_dir, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
