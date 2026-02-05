#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIG ----
SRC="/home/adamranson/temp/speed_test/combined_stack_0.tif"
DEST_USER="uab586750"
DEST_HOST="hsmtransfer1.bsc.es"
DEST_DIR="/gpfs/archive/resdata/res23025"

INTERVAL_SECONDS=$((1*60))   # 1 minute
LOG_DIR="$HOME/rsync_speed_logs"
SPEEDS_FILE="$LOG_DIR/speeds_MBps.tsv"
ATTEMPTS_LOG="$LOG_DIR/attempts.log"
# --------------

# Prompt once for password (no echo)
read -rsp "SSH password for ${DEST_USER}@${DEST_HOST}: " SSH_PASSWORD
echo

# Require tools
command -v sshpass >/dev/null 2>&1 || { echo "sshpass not found. Install: sudo apt-get install -y sshpass"; exit 127; }
command -v rsync >/dev/null 2>&1 || { echo "rsync not found. Install: sudo apt-get install -y rsync"; exit 127; }
command -v ssh >/dev/null 2>&1 || { echo "ssh not found"; exit 127; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; exit 127; }
[[ -x /usr/bin/time ]] || { echo "/usr/bin/time not found or not executable"; exit 127; }

mkdir -p "$LOG_DIR"
touch "$SPEEDS_FILE" "$ATTEMPTS_LOG"

bytes_to_MB() {
  python3 - <<PY
b=float("$1")
print(b/1e6)
PY
}

mean() {
  awk '{sum+=$2; n+=1} END{ if(n>0) printf "%.3f", sum/n; else print "0" }' "$SPEEDS_FILE"
}

print_summary() {
  echo "---- Attempts so far ----"
  if [[ -s "$SPEEDS_FILE" ]]; then
    awk '{printf "Attempt %d: %.3f MB/s\n", NR, $2}' "$SPEEDS_FILE"
    echo "Mean: $(mean) MB/s"
  else
    echo "(none)"
  fi
  echo "-------------------------"
}

SSH_OPTS="-o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new"
RSYNC_SSH="ssh $SSH_OPTS"

while true; do
  ts="$(date -Is)"
  echo "[$ts] Starting attempt..." | tee -a "$ATTEMPTS_LOG"

  if [[ ! -f "$SRC" ]]; then
    echo "[$ts] Source file not found: $SRC" | tee -a "$ATTEMPTS_LOG"
    print_summary
    echo "Sleeping..."
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  base="$(basename "$SRC")"
  remote_path="${DEST_DIR%/}/$base"

  size_bytes="$(stat -c '%s' "$SRC")"
  size_MB="$(bytes_to_MB "$size_bytes")"

  tmp_time="$(mktemp)"

  set +e
  # rsync progress (-P) + numeric speeds (--info=progress2)
  sshpass -p "$SSH_PASSWORD" \
    /usr/bin/time -f "REAL_SECONDS=%e" -o "$tmp_time" \
    rsync -a --no-compress -P --info=progress2 \
      -e "$RSYNC_SSH" \
      "$SRC" "${DEST_USER}@${DEST_HOST}:${DEST_DIR%/}/"
  rc=$?
  set -e

  real_seconds="$(grep -oE 'REAL_SECONDS=[0-9.]+' "$tmp_time" | cut -d= -f2 || true)"
  rm -f "$tmp_time"

  if [[ $rc -ne 0 || -z "${real_seconds:-}" ]]; then
    echo "[$ts] Upload FAILED (rc=$rc)" | tee -a "$ATTEMPTS_LOG"
    print_summary
    echo "Sleeping..."
    sleep "$INTERVAL_SECONDS"
    continue
  fi

  speed_MBps="$(python3 - <<PY
mb=float("$size_MB")
secs=float("$real_seconds")
print("0" if secs<=0 else f"{mb/secs:.3f}")
PY
)"

  echo -e "${ts}\t${speed_MBps}" >> "$SPEEDS_FILE"
  echo "[$ts] Upload OK. Size=${size_MB} MB, Time=${real_seconds}s, Speed=${speed_MBps} MB/s" | tee -a "$ATTEMPTS_LOG"

  # Delete DESTINATION (remote) file, NOT the source
  sshpass -p "$SSH_PASSWORD" \
    ssh $SSH_OPTS "${DEST_USER}@${DEST_HOST}" \
    "rm -f -- '$remote_path'"
  echo "[$ts] Deleted remote file: $remote_path" | tee -a "$ATTEMPTS_LOG"

  print_summary
  echo "Sleeping..."
  sleep "$INTERVAL_SECONDS"
done
