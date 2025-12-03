#!/usr/bin/env bash
set -euo pipefail

# start_service.sh
# Start 8 uvicorn EmbeddingService instances:
#  - ports 8090-8093 -> pinned to CUDA_VISIBLE_DEVICES=0 (physical GPU 0)
#  - ports 8094-8097 -> pinned to CUDA_VISIBLE_DEVICES=1 (physical GPU 1)
# Writes logs to ./logs/uvicorn_<port>.log and pids to ./run/pid_<port>.txt

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
RUN_DIR="$ROOT_DIR/run"
mkdir -p "$LOG_DIR" "$RUN_DIR"

ports=(8090 8091 8092 8093 8094 8095 8096 8097)

echo "Starting ${#ports[@]} services..."
for i in "${!ports[@]}"; do
  port=${ports[$i]}
  if [ "$i" -lt 4 ]; then
    gpu=0
  else
    gpu=1
  fi

  logfile="$LOG_DIR/uvicorn_${port}.log"
  pidfile="$RUN_DIR/pid_${port}.txt"

  echo "- Starting port $port on CUDA_VISIBLE_DEVICES=$gpu -> log: $logfile"

  # Start with nohup so the process survives the shell session
  ( env CUDA_VISIBLE_DEVICES="$gpu" nohup uvicorn src.service.main:app --host 0.0.0.0 --port "$port" --log-level info > "$logfile" 2>&1 & echo $! > "$pidfile" )

  # small stagger so logs and pid files are created sequentially
  sleep 0.5
done

echo
echo "All start commands issued. Polling /health/ready for each instance (timeout 60s)..."

timeout_seconds=60
start_time=$SECONDS

for port in "${ports[@]}"; do
  echo -n "Waiting for port $port: "
  deadline=$(( start_time + timeout_seconds ))
  ready=false
  while [ $SECONDS -lt $deadline ]; do
    if curl -sS -m 3 "http://127.0.0.1:$port/health/ready" >/dev/null 2>&1; then
      echo "READY"
      ready=true
      break
    else
      echo -n "."
      sleep 1
    fi
  done
  if [ "$ready" = false ]; then
    echo " FAILED (see $LOG_DIR/uvicorn_${port}.log)"
  fi
done

echo
echo "Started services and PID files:"
for port in "${ports[@]}"; do
  pidfile="$RUN_DIR/pid_${port}.txt"
  pid="$(cat "$pidfile" 2>/dev/null || echo 'no pid')"
  printf "port %s -> %s\n" "$port" "$pid"
done

echo
echo "To inspect logs, run: tail -n 200 logs/uvicorn_<port>.log"
echo "To stop a service: kill \\$(cat run/pid_<port>.txt)"
