from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from benchmark_torchscript import DEFAULT_MODELS_DIR, WORKSPACE_DIR, resolve_device


DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "artifacts" / "onnx" / "loma_b128_1024_k2048"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将导出的 LoMa TorchScript 三段模型导出为固定 shape ONNX。"
    )
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--num-keypoints", type=int, default=2048)
    parser.add_argument("--descriptor-dim", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto"
    )
    parser.add_argument(
        "--skip-checker",
        action="store_true",
        help="跳过 ONNX 模型检查；仅用于排查导出工具问题。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="明确允许覆盖 output-dir 中已有的 ONNX 文件。",
    )
    return parser.parse_args()


def require_onnx() -> object:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "当前 Python 环境缺少 onnx。请先在执行导出的同一 Conda 环境中安装"
            "与 PyTorch/TensorRT 兼容的 ONNX 包，再重新执行。"
        ) from error
    return onnx


def load_module(path: Path, device: torch.device) -> torch.jit.ScriptModule:
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型文件: {path}")
    return torch.jit.load(str(path), map_location=device).eval()


def export_module(
    module: torch.jit.ScriptModule,
    inputs: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    opset: int,
) -> None:
    torch.onnx.export(
        module,
        inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=None,
        keep_initializers_as_inputs=False,
        dynamo=False,
    )


def main() -> None:
    args = parse_args()
    if args.input_size <= 0:
        raise ValueError("input-size 必须大于零。")
    if args.num_keypoints <= 0:
        raise ValueError("num-keypoints 必须大于零。")
    if args.descriptor_dim <= 0:
        raise ValueError("descriptor-dim 必须大于零。")

    onnx = require_onnx()
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    outputs = {
        "detector": output_dir / "loma_detector.onnx",
        "descriptor": output_dir / "loma_descriptor.onnx",
        "matcher": output_dir / "loma_matcher.onnx",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        formatted_paths = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "检测到已有 ONNX 文件；请更换 output-dir，或确认后使用 --overwrite：\n"
            f"{formatted_paths}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    image = torch.rand(
        1, 3, args.input_size, args.input_size, dtype=torch.float32, device=device
    )
    keypoints_a = torch.rand(
        args.num_keypoints, 2, dtype=torch.float32, device=device
    ).mul_(2).sub_(1)
    keypoints_b = torch.rand(
        args.num_keypoints, 2, dtype=torch.float32, device=device
    ).mul_(2).sub_(1)
    descriptors_a = torch.rand(
        args.num_keypoints,
        args.descriptor_dim,
        dtype=torch.float32,
        device=device,
    )
    descriptors_b = torch.rand(
        args.num_keypoints,
        args.descriptor_dim,
        dtype=torch.float32,
        device=device,
    )

    detector = load_module(args.models_dir / "loma_detector.pt", device)
    descriptor = load_module(args.models_dir / "loma_descriptor.pt", device)
    matcher = load_module(args.models_dir / "loma_matcher.pt", device)

    export_module(
        detector,
        (image,),
        outputs["detector"],
        ["image"],
        ["keypoints", "keypoint_probs"],
        args.opset,
    )
    export_module(
        descriptor,
        (image, keypoints_a),
        outputs["descriptor"],
        ["image", "keypoints"],
        ["descriptors"],
        args.opset,
    )
    export_module(
        matcher,
        (keypoints_a, descriptors_a, keypoints_b, descriptors_b),
        outputs["matcher"],
        ["keypoints0", "descriptors0", "keypoints1", "descriptors1"],
        ["matches0", "match_scores0"],
        args.opset,
    )

    if not args.skip_checker:
        for output_path in outputs.values():
            onnx.checker.check_model(onnx.load(str(output_path)))

    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_models_dir": str(args.models_dir.resolve()),
        "device": str(device),
        "opset": args.opset,
        "static_shapes": {
            "image": [1, 3, args.input_size, args.input_size],
            "keypoints": [args.num_keypoints, 2],
            "descriptors": [args.num_keypoints, args.descriptor_dim],
        },
        "models": {
            name: {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }
    manifest_path = output_dir / "export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"ONNX 导出完成: {output_dir}")


if __name__ == "__main__":
    main()
