#!/usr/bin/env bash
# Reference load-client call for srv-bench. THIS FILE IS PART OF THE HARNESS: edit it.
#
#   bash launch_kit/aisbench_example.sh perf 4096 2048
#   bash launch_kit/aisbench_example.sh perf 16384 1024
#   bash launch_kit/aisbench_example.sh accuracy
#
# perf     reports throughput, TTFT and TPOT for one (input, output) length pair
# accuracy runs the GSM8K prompt set the image ships with
set -euo pipefail

MODE="${1:-perf}"
IN_LEN="${2:-4096}"
OUT_LEN="${3:-2048}"
CONCURRENCY="${CONCURRENCY:-64}"
DATASET="${DATASET:-gsm8k_gen_0_shot_cot_chat_prompt.py}"
OUT_DIR="${OUT_DIR:-./artifacts}"

mkdir -p "$OUT_DIR"

case "$MODE" in
  perf)
    # TODO: check every flag against the ais_bench inside SRV_IMAGE -- "ais_bench --help" is
    # the authority, this file is a starting point.
    ais_bench --models vllm_api_stream_chat \
      --datasets synthetic_gen --mode perf \
      --input-len "$IN_LEN" --output-len "$OUT_LEN" \
      --num-prompts "$CONCURRENCY" --max-concurrency "$CONCURRENCY" \
      --output-dir "$OUT_DIR/perf_${IN_LEN}_${OUT_LEN}"
    ;;
  accuracy)
    ais_bench --models vllm_api_stream_chat \
      --datasets "$DATASET" --mode accuracy \
      --output-dir "$OUT_DIR/accuracy"
    ;;
  *)
    echo "usage: $0 [perf <input_len> <output_len>|accuracy]" >&2
    exit 2
    ;;
esac
