from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from benchmark_torchscript import (
    DEFAULT_MODELS_DIR,
    WORKSPACE_DIR,
    TorchScriptLoMa,
    capture_peak_memory,
    load_letterboxed_image,
    make_pipeline_operation,
    match_statistics,
    resolve_device,
    benchmark,
)


DEFAULT_DATA_DIR = WORKSPACE_DIR / "data"
DEFAULT_RESULTS_DIR = WORKSPACE_DIR / "results" / "torchscript_suite"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="遍历 ws/data 中的 base.jpeg 图像对并生成 TorchScript 基线报告。"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="默认关闭，作为 F32 数值参考基线。",
    )
    parser.add_argument("--ransac-threshold", type=float, default=0.5)
    parser.add_argument("--ransac-confidence", type=float, default=0.999999)
    parser.add_argument("--ransac-max-iters", type=int, default=10000)
    return parser.parse_args()


def discover_pairs(data_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"找不到数据目录: {data_dir}")

    pairs: list[tuple[Path, Path]] = []
    warnings: list[str] = []
    for base_path in sorted(data_dir.rglob("base.jpeg")):
        candidates = sorted(
            path
            for path in base_path.parent.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.name != base_path.name
        )
        if len(candidates) != 1:
            warnings.append(
                f"跳过 {base_path.parent}: 期望 1 张配对图像，实际找到 {len(candidates)} 张。"
            )
            continue
        pairs.append((base_path, candidates[0]))
    return pairs, warnings


def write_result(results_dir: Path, result: dict[str, object]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"torchscript_suite_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2, ensure_ascii=False)
    return output_path


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    pairs, warnings = discover_pairs(args.data_dir)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit 必须大于零。")
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("未发现可测试图像对。")

    model = TorchScriptLoMa(args.models_dir, device)
    case_results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for index, (base_path, target_path) in enumerate(pairs, start=1):
        case_name = str(base_path.parent.relative_to(args.data_dir))
        print(f"[{index}/{len(pairs)}] {case_name}")
        try:
            image_a, transform_a = load_letterboxed_image(
                base_path, args.input_size, device
            )
            image_b, transform_b = load_letterboxed_image(
                target_path, args.input_size, device
            )
            operation = make_pipeline_operation(model, image_a, image_b)
            _, timing = benchmark(operation, args.warmup, args.iterations, device)
            output, memory = capture_peak_memory(operation, device)
            matching = match_statistics(
                output,
                transform_a,
                transform_b,
                args.ransac_threshold,
                args.ransac_confidence,
                args.ransac_max_iters,
            )
            case_result = {
                "case": case_name,
                "image_a": str(base_path.resolve()),
                "image_b": str(target_path.resolve()),
                "timing_ms": timing,
                "memory_mib": memory,
                "matching": matching,
            }
            case_results.append(case_result)
            print(
                "  "
                f"P50={timing['p50_ms']} ms, "
                f"P95={timing['p95_ms']} ms, "
                f"inliers={matching['ransac']['inliers']}"
            )
        except Exception as error:
            errors.append({"case": case_name, "error": str(error)})
            print(f"  失败: {error}")

    p50_values = [item["timing_ms"]["p50_ms"] for item in case_results]
    p95_values = [item["timing_ms"]["p95_ms"] for item in case_results]
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
        "config": {
            "data_dir": str(args.data_dir.resolve()),
            "models_dir": str(args.models_dir.resolve()),
            "input_size": args.input_size,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "ransac_threshold": args.ransac_threshold,
        },
        "summary": {
            "discovered_cases": len(pairs),
            "completed_cases": len(case_results),
            "failed_cases": len(errors),
            "mean_case_p50_ms": round(sum(p50_values) / len(p50_values), 3)
            if p50_values
            else None,
            "mean_case_p95_ms": round(sum(p95_values) / len(p95_values), 3)
            if p95_values
            else None,
            "worst_case_p95_ms": round(max(p95_values), 3) if p95_values else None,
        },
        "warnings": warnings,
        "errors": errors,
        "cases": case_results,
    }
    output_path = write_result(args.results_dir, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"结果已保存: {output_path}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
