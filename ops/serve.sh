#!/bin/sh
# Keep a ComfyUI and the Comfy API v2 proxy (comfy-api-proxy) running, restarting either if it exits.
#   ops/serve.sh start | stop | status
# Env: COMFYUI_DIR (default ~/ComfyUI), COMFYUI_PY (default $COMFYUI_DIR/venv/bin/python), COMFY_PORT (8198),
#      PROXY_PORT (8195), COMFY_ARGS (extra ComfyUI flags), PROXY_BIN (default .venv/bin/comfy-api-proxy).
# Point the runner and the app at the proxy: COMFY_BASE_URL=http://127.0.0.1:8195
set -eu
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
ROOT=$(cd "$(dirname "$0")/.." && pwd)
COMFYUI_DIR=${COMFYUI_DIR:-$HOME/ComfyUI}
COMFYUI_PY=${COMFYUI_PY:-$COMFYUI_DIR/venv/bin/python}
COMFY_PORT=${COMFY_PORT:-8198}
PROXY_PORT=${PROXY_PORT:-8195}
COMFY_ARGS=${COMFY_ARGS:---bf16-unet --cpu-vae}
PROXY_BIN=${PROXY_BIN:-$ROOT/.venv/bin/comfy-api-proxy}
RUN=$ROOT/.state/serve
mkdir -p "$RUN" "$ROOT/.comfy/output"

case ${1:-status} in
  _supervise) # internal: _supervise <name> <workdir> <command...>
    name=$2; dir=$3; shift 3
    cd "$dir"
    while [ -f "$RUN/enabled" ]; do
      echo "[$(date '+%F %T')] starting $name" >> "$RUN/$name.log"
      "$@" >> "$RUN/$name.log" 2>&1 &
      echo $! > "$RUN/$name.pid"
      rc=0; wait $! || rc=$?
      echo "[$(date '+%F %T')] $name exited rc=$rc" >> "$RUN/$name.log"
      sleep 5
    done
    ;;
  start)
    touch "$RUN/enabled"
    # shellcheck disable=SC2086
    nohup "$SELF" _supervise comfyui "$COMFYUI_DIR" "$COMFYUI_PY" main.py --listen 127.0.0.1 --port "$COMFY_PORT" \
      --disable-auto-launch $COMFY_ARGS --output-directory "$ROOT/.comfy/output" > /dev/null 2>&1 &
    echo $! > "$RUN/comfyui.sup"
    nohup "$SELF" _supervise proxy "$ROOT" "$PROXY_BIN" run --comfyui "http://127.0.0.1:$COMFY_PORT" \
      --port "$PROXY_PORT" --comfyui-base-dir "$COMFYUI_DIR" --state-dir "$ROOT/.state" --max-upload-mb 500 > /dev/null 2>&1 &
    echo $! > "$RUN/proxy.sup"
    echo "started: ComfyUI :$COMFY_PORT, API v2 proxy :$PROXY_PORT (logs in $RUN)"
    ;;
  stop)
    rm -f "$RUN/enabled"
    for n in comfyui proxy; do
      [ -f "$RUN/$n.sup" ] && kill "$(cat "$RUN/$n.sup")" 2>/dev/null || true
      [ -f "$RUN/$n.pid" ] && kill "$(cat "$RUN/$n.pid")" 2>/dev/null || true
    done
    echo stopped
    ;;
  status)
    if curl -s -m3 "http://127.0.0.1:$COMFY_PORT/system_stats" > /dev/null; then echo "ComfyUI :$COMFY_PORT up"; else echo "ComfyUI :$COMFY_PORT down"; fi
    curl -s -m3 "http://127.0.0.1:$PROXY_PORT/api/v2/health" || echo "proxy :$PROXY_PORT down"
    echo
    ;;
esac
