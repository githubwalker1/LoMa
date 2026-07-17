#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ONNX_DIR="${WORKSPACE_DIR}/artifacts/onnx/loma_b128_1024_k2048"
ENGINE_DIR="${WORKSPACE_DIR}/engines/loma_b128_1024_k2048/fp16"
WORKSPACE_MIB=1024
TRTEXEC_BIN="${TRTEXEC_BIN:-trtexec}"

usage() {
  cat <<'EOF'
用法：
  bash ws/scripts/build_tensorrt_fp16.sh [选项]

选项：
  --onnx-dir PATH       ONNX 文件目录
  --engine-dir PATH     FP16 engine 输出目录
  --workspace-mib SIZE  TensorRT workspace 上限，默认 1024 MiB
  --trtexec PATH        trtexec 可执行文件路径
  -h, --help            显示帮助

要求：
  - 在目标 Orin NX 上执行；生成的 engine 不可跨 GPU 架构复用。
  - ONNX 目录必须包含 export_manifest.json 和三个 loma_*.onnx 文件。
  - 输出目录存在同名 engine 时脚本会失败，防止意外覆盖。
EOF
}

while (($# > 0)); do
  case "$1" in
    --onnx-dir)
      ONNX_DIR="$2"
      shift 2
      ;;
    --engine-dir)
      ENGINE_DIR="$2"
      shift 2
      ;;
    --workspace-mib)
      WORKSPACE_MIB="$2"
      shift 2
      ;;
    --trtexec)
      TRTEXEC_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知选项: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$WORKSPACE_MIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "workspace-mib 必须是正整数。" >&2
  exit 2
fi
if ! command -v "$TRTEXEC_BIN" >/dev/null 2>&1; then
  echo "找不到 trtexec: $TRTEXEC_BIN" >&2
  exit 1
fi
if [[ ! -f "${ONNX_DIR}/export_manifest.json" ]]; then
  echo "找不到 ONNX 导出清单: ${ONNX_DIR}/export_manifest.json" >&2
  exit 1
fi

mkdir -p "$ENGINE_DIR" "${ENGINE_DIR}/logs" "${ENGINE_DIR}/timing_cache"

build_engine() {
  local name="$1"
  local onnx_path="${ONNX_DIR}/loma_${name}.onnx"
  local engine_path="${ENGINE_DIR}/loma_${name}.engine"
  local log_path="${ENGINE_DIR}/logs/loma_${name}_build.log"
  local layer_info_path="${ENGINE_DIR}/logs/loma_${name}_layers.json"
  local timing_cache_path="${ENGINE_DIR}/timing_cache/loma_${name}.cache"

  if [[ ! -f "$onnx_path" ]]; then
    echo "找不到 ONNX 文件: $onnx_path" >&2
    exit 1
  fi
  if [[ -e "$engine_path" ]]; then
    echo "目标 engine 已存在，拒绝覆盖: $engine_path" >&2
    exit 1
  fi

  echo "构建 FP16 engine: $name"
  "$TRTEXEC_BIN" \
    --onnx="$onnx_path" \
    --saveEngine="$engine_path" \
    --fp16 \
    --memPoolSize="workspace:${WORKSPACE_MIB}" \
    --builderOptimizationLevel=5 \
    --profilingVerbosity=detailed \
    --exportLayerInfo="$layer_info_path" \
    --timingCacheFile="$timing_cache_path" \
    --skipInference \
    2>&1 | tee "$log_path"
}

build_engine "detector"
build_engine "descriptor"
build_engine "matcher"

echo "FP16 TensorRT engine 构建完成: $ENGINE_DIR"
