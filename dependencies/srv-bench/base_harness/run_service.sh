#!/usr/bin/env bash
# Reference service launcher for srv-bench. THIS FILE IS PART OF THE HARNESS: edit it.
#
#   bash launch_kit/run_service.sh single   # one instance
#   bash launch_kit/run_service.sh pd       # one prefill + one decode instance, one container
#
# Environment it reads:
#   SRV_IMAGE   the vllm-ascend image this host runs (required; never hardcode a tag)
#   MODEL       default Qwen/Qwen3-8B
#   MAX_LEN     default 32768 (the model's native context; YaRN stays off)
#   TP          default 8 (tensor-parallel size)
#   PORT_P/PORT_D  default 8000 / 8001
#   LOG_DIR     default ./logs
set -euo pipefail

MODE="${1:-single}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
MAX_LEN="${MAX_LEN:-32768}"
TP="${TP:-8}"
PORT_P="${PORT_P:-8000}"
PORT_D="${PORT_D:-8001}"
LOG_DIR="${LOG_DIR:-./logs}"

: "${SRV_IMAGE:?set SRV_IMAGE to the vllm-ascend image for this host}"
mkdir -p "$LOG_DIR"

common_args=(
  --model "$MODEL"
  --max-model-len "$MAX_LEN"
  --tensor-parallel-size "$TP"
  --trust-remote-code
)

wait_healthy() {
  local port="$1" name="$2"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "[${name}] healthy on ${port}"
      return 0
    fi
    sleep 2
  done
  echo "[${name}] did NOT become healthy; read ${LOG_DIR}/${name}.log" >&2
  return 1
}

case "$MODE" in
  single)
    # TODO: put the launch flags this assignment asks you to compare here, one change at a
    # time, and keep the same client call between two attempts.
    python -m vllm.entrypoints.openai.api_server "${common_args[@]}" \
      --port "$PORT_P" --enable-prefix-caching \
      >"${LOG_DIR}/p.log" 2>&1 &
    wait_healthy "$PORT_P" p
    ;;
  pd)
    # 1P1D inside one container: prefill keeps the prefix cache, decode turns it off.
    # TODO: add the KV-transfer connector this image supports, for example
    #   --kv-transfer-config '{"kv_connector":"AscendConnector","kv_role":"kv_producer"}'
    # on the prefill side and "kv_consumer" on the decode side.
    python -m vllm.entrypoints.openai.api_server "${common_args[@]}" \
      --port "$PORT_P" --enable-prefix-caching --gpu-memory-utilization 0.5 \
      >"${LOG_DIR}/p.log" 2>&1 &
    python -m vllm.entrypoints.openai.api_server "${common_args[@]}" \
      --port "$PORT_D" --no-enable-prefix-caching --gpu-memory-utilization 0.5 \
      >"${LOG_DIR}/d.log" 2>&1 &
    wait_healthy "$PORT_P" p
    wait_healthy "$PORT_D" d
    ;;
  *)
    echo "usage: $0 [single|pd]" >&2
    exit 2
    ;;
esac

echo "service up: mode=${MODE} model=${MODEL} max_len=${MAX_LEN} tp=${TP}"
echo "prefill on http://127.0.0.1:${PORT_P}"
