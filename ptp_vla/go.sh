#!/usr/bin/env bash
# One entry point: stops leftovers, reuses or creates the environment, then runs with automatic restart.
# Start it through `bash go.sh`. It moves itself to the background and shows the log.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
LOG="$PWD/run.log"

if [ "${1:-}" != "--worker" ]; then
  MODE=${1:-all}
  pkill -f '[r]un\.py (all|quick)' 2>/dev/null
  pkill -9 -f 'multiprocessing\.spawn|multiprocessing\.forkserver|multiprocessing\.resource_tracker' 2>/dev/null
  sleep 3
  nohup bash "$0" --worker "$MODE" >> "$LOG" 2>&1 &
  echo "Started. Log: $LOG"
  sleep 2
  exec tail -n 5 -f "$LOG"
fi

echo "=== go.sh started $(date)"
if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import lerobot, libero, torch" </dev/null 2>/dev/null; then
  echo "--- no working environment found, installing"
  bash setup.sh || { echo "SETUP FAILED"; exit 1; }
fi
# shellcheck disable=SC1091
source .venv/bin/activate
[ -f .render_backend ] || bash fix_render.sh || { echo "RENDER FAILED"; exit 1; }
export TQDM_DISABLE=1
echo "--- machine"
free -g | head -2
cat /sys/fs/cgroup/memory.max 2>/dev/null | sed 's/^/container memory limit: /'
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

for attempt in 1 2 3 4 5 6; do
  echo "=== run attempt $attempt $(date)"
  python -u run.py "${2:-all}" 2>&1 | grep --line-buffered -v -E 'robosuite|Local assets not found|Assets already downloaded'
  code=${PIPESTATUS[0]}
  if [ "$code" -eq 0 ]; then
    echo ALL_DONE
    exit 0
  fi
  echo "=== run stopped with exit code $code at $(date). 137 means the system killed it, usually for memory."
  dmesg 2>/dev/null | grep -i -E 'killed process|out of memory' | tail -3
  free -g | head -2
  pkill -9 -f 'multiprocessing\.spawn|multiprocessing\.forkserver|multiprocessing\.resource_tracker' 2>/dev/null
  sleep 15
done
echo "GAVE UP after 6 attempts"
exit 1
