#!/usr/bin/env bash
set -euo pipefail

campaign="${1:?campaign is required}"
seed="${2:?seed is required}"
source_checkpoint="${3:?remote source checkpoint is required}"
source_sha256="${4:?source SHA-256 is required}"
session="${5:-phase12-filtered}"
start_fresh="${6:-0}"
recovery_run="${7:-0}"

export HTTP_PROXY="socks5h://127.0.0.1:10808"
export HTTPS_PROXY="socks5h://127.0.0.1:10808"
export ALL_PROXY="socks5h://127.0.0.1:10808"

colab_cli="${HOME}/.local/bin/colab"
project_root="/mnt/f/Projects/Steel Frame Optimization"

"${colab_cli}" status --session "${session}"
"${colab_cli}" exec \
  --session "${session}" \
  --file "${project_root}/deployment/colab_phase12_seed_job.py" \
  --env "PHASE12_CAMPAIGN=${campaign}" \
  --env "PHASE12_SEED=${seed}" \
  --env "PHASE12_SOURCE_CHECKPOINT=${source_checkpoint}" \
  --env "PHASE12_SOURCE_SHA256=${source_sha256}" \
  --env "PHASE12_START_FRESH=${start_fresh}" \
  --env "PHASE12_RECOVERY_RUN=${recovery_run}" \
  --env "PHASE12_RECOVERY_MANIFEST=/content/phase12_recovery_manifest.json" \
  --env "PHASE12_DRIVE_ROOT=/content/drive/MyDrive/steel-frame-phase12-v1/campaigns" \
  --timeout 86400
