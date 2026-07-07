from __future__ import annotations

import argparse
from pathlib import Path

import torch

from loma import LoMa, LoMaB128, LoMaB, LoMaG, LoMaL, LoMaR
from loma.loma import filter_matches


MODEL_PRESETS = {
    "loma_b128": LoMaB128,
    "loma_b": LoMaB,
    "loma_l": LoMaL,
    "loma_g": LoMaG,
    "loma_r": LoMaR,
}


class DetectDescribeWrapper(torch.nn.Module):
    def __init__(self, model: LoMa, num_keypoints: int) -> None:
        super().__init__()
        self.model = model
        self.num_keypoints = num_keypoints

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keypoints, descriptors, _, _ = self.model.detect_and_describe(
            image,
            num_keypoints=self.num_keypoints,
        )
        return keypoints[0].float(), descriptors[0].float()


class MatcherWrapper(torch.nn.Module):
    def __init__(self, model: LoMa, threshold: float) -> None:
        super().__init__()
        self.model = model
        self.threshold = threshold

    def forward(
        self,
        keypoints0: torch.Tensor,
        descriptors0: torch.Tensor,
        keypoints1: torch.Tensor,
        descriptors1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.model(
            keypoints0,
            keypoints1,
            descriptors0,
            descriptors1,
        )["scores"]
        matches0, _, match_scores0, _ = filter_matches(scores, self.threshold)
        return matches0[0], match_scores0[0].float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LoMa TorchScript modules for ORB-SLAM3 C++ integration.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_PRESETS.keys(), default="loma_b128")
    parser.add_argument("--num-keypoints", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--match-threshold", type=float, default=0.1)
    parser.add_argument(
        "--kind",
        choices=("detect_describe", "matcher", "all"),
        default="all",
        help="Which TorchScript module to export.",
    )
    return parser.parse_args()


def export_detect_describe(model: LoMa, args: argparse.Namespace) -> None:
    wrapper = DetectDescribeWrapper(model, args.num_keypoints).eval()
    example = torch.rand(1, 3, args.image_size, args.image_size, device=next(model.parameters()).device)
    traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
    traced.save(str(args.output_dir / "loma_detect_describe.pt"))


def export_matcher(model: LoMa, args: argparse.Namespace) -> None:
    wrapper = MatcherWrapper(model, args.match_threshold).eval()
    device = next(model.parameters()).device
    descriptor_dim = model.cfg.input_dim
    keypoints0 = torch.rand(1, args.num_keypoints, 2, device=device) * 2 - 1
    keypoints1 = torch.rand(1, args.num_keypoints, 2, device=device) * 2 - 1
    descriptors0 = torch.rand(1, args.num_keypoints, descriptor_dim, device=device)
    descriptors1 = torch.rand(1, args.num_keypoints, descriptor_dim, device=device)
    traced = torch.jit.trace(
        wrapper,
        (keypoints0, descriptors0, keypoints1, descriptors1),
        strict=False,
        check_trace=False,
    )
    traced.save(str(args.output_dir / "loma_matcher.pt"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = LoMa(MODEL_PRESETS[args.model]()).eval()
    with torch.inference_mode():
        if args.kind in ("detect_describe", "all"):
            export_detect_describe(model, args)
        if args.kind in ("matcher", "all"):
            export_matcher(model, args)


if __name__ == "__main__":
    main()
