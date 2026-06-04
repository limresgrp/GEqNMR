#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DATA_ROOT="${ROOT_DIR}/outputs"
ENV_FILE="${ROOT_DIR}/.env"
DEFAULT_PYTHON="${ROOT_DIR}/.venv-geqnmr/bin/python"

read_env_value() {
  local key="$1"
  if [ ! -f "$ENV_FILE" ]; then
    return 1
  fi
  python - "$ENV_FILE" "$key" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
for raw in path.read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    name, value = line.split("=", 1)
    if name.strip() != key:
        continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1]
    print(value)
    sys.exit(0)
sys.exit(1)
PY
}

SAVED_DATA_ROOT="$(read_env_value GEQNMR_DATA_ROOT || true)"
DATA_ROOT="${GEQNMR_DATA_ROOT:-${SAVED_DATA_ROOT:-${DEFAULT_DATA_ROOT}}}"

ensure_data_root() {
  mkdir -p "${DATA_ROOT}/prepared_inputs"
  chmod 755 "${DATA_ROOT}" "${DATA_ROOT}/prepared_inputs" 2>/dev/null || true
  export GEQNMR_DATA_ROOT="${DATA_ROOT}"
}

write_env_value() {
  local key="$1"
  local value="$2"
  python - "$ENV_FILE" "$key" "$value" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text().splitlines() if path.exists() else []
updated = False
next_lines = []
for line in lines:
    stripped = line.strip()
    if stripped and not stripped.startswith("#") and "=" in line:
        name = line.split("=", 1)[0].strip()
        if name == key:
            next_lines.append(f"{key}={value}")
            updated = True
            continue
    next_lines.append(line)
if not updated:
    next_lines.append(f"{key}={value}")
path.write_text("\n".join(next_lines) + "\n")
PY
  chmod 644 "$ENV_FILE" 2>/dev/null || true
}

ensure_data_root

prompt() {
  local label="$1"
  local default="$2"
  local value
  read -r -p "${label} [${default}]: " value
  if [ -z "$value" ]; then
    echo "$default"
  else
    echo "$value"
  fi
}

run_cli() {
  local python_bin="${GEQNMR_PYTHON:-python}"
  if [ -z "${GEQNMR_PYTHON:-}" ] && [ -x "$DEFAULT_PYTHON" ]; then
    python_bin="$DEFAULT_PYTHON"
  fi
  "$python_bin" -m backend.app.cli "$@"
}

change_root() {
  local next_root
  next_root="$(prompt "Shared data root" "$DATA_ROOT")"
  if [ -z "$next_root" ]; then
    return
  fi
  next_root="$(
    python - "$next_root" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
  )"
  DATA_ROOT="$next_root"
  ensure_data_root
  write_env_value GEQNMR_DATA_ROOT "$DATA_ROOT"
  echo "Console root set to: ${DATA_ROOT}"
  echo "Saved to: ${ENV_FILE}"
  echo "Restart the web backend to apply it:"
  echo "  docker compose up -d backend"
}

choose_from_json_list() {
  local json="$1"
  local key="$2"
  JSON_PAYLOAD="$json" python - "$key" <<'PY'
import json, os, sys
key = sys.argv[1]
data = json.loads(os.environ["JSON_PAYLOAD"])
items = data.get(key, [])
for idx, item in enumerate(items, 1):
    label = item.get("name") or item.get("id") or item.get("model") or str(item)
    extra = item.get("input_file") or item.get("size_bytes") or ""
    print(f"{idx}|{item.get('id', item.get('name', label))}|{label}|{extra}")
PY
}

print_rows() {
  local rows="$1"
  if [ -z "$rows" ]; then
    echo "  (none)"
    return
  fi
  while IFS='|' read -r idx id label extra; do
    [ -z "${idx:-}" ] && continue
    echo "  ${idx}) ${label} (${id})"
    [ -n "${extra:-}" ] && echo "      ${extra}"
  done <<< "$rows"
}

select_row_id() {
  local label="$1"
  local rows="$2"
  if [ -z "$rows" ]; then
    return 1
  fi
  echo "$label:" >&2
  print_rows "$rows" >&2
  local choice
  printf "Choice [1]: " >&2
  read -r choice
  if [ -z "$choice" ]; then
    choice="1"
  fi
  awk -F'|' -v choice="$choice" '$1 == choice {print $2; found=1} END {exit found ? 0 : 1}' <<< "$rows"
}

list_prepared() {
  local json rows
  json="$(run_cli list-prepared)"
  rows="$(choose_from_json_list "$json" prepared)"
  echo "Prepared inputs:"
  print_rows "$rows"
}

list_results() {
  local json
  json="$(run_cli list-results)"
  JSON_PAYLOAD="$json" python - <<'PY'
import json, datetime, os
data = json.loads(os.environ["JSON_PAYLOAD"])
items = data.get("results", [])
if not items:
    print("Results:\n  (none)")
else:
    print("Results:")
    for item in items:
        ts = datetime.datetime.fromtimestamp(item.get("modified", 0)).isoformat(sep=" ", timespec="seconds")
        marker = " prediction-metadata" if item.get("has_predictions") else ""
        print(f"  - {item['name']} ({item.get('size_bytes', 0)} bytes, {ts}){marker}")
PY
}

add_prepared() {
  local file traj name num_workers
  local -a command
  file="$(prompt "Input file path" "")"
  if [ ! -f "$file" ]; then
    echo "File not found: $file"
    return
  fi
  traj="$(prompt "Trajectory file path (blank for none)" "")"
  if [ -n "$traj" ] && [ ! -f "$traj" ]; then
    echo "Trajectory file not found: $traj"
    return
  fi
  name="$(prompt "Stored input name" "$(basename "$file")")"
  num_workers="$(prompt "Input processing workers" "8")"

  command=(prepare --input "$file" --name "$name" --workers "$num_workers")
  if [ -n "$traj" ]; then
    command+=(--trajectory "$traj")
  fi
  run_cli "${command[@]}"
}

delete_prepared() {
  local json rows id
  json="$(run_cli list-prepared)"
  rows="$(choose_from_json_list "$json" prepared)"
  if ! id="$(select_row_id "Delete prepared input number" "$rows")"; then
    echo "Invalid selection."
    return
  fi
  run_cli delete-prepared "$id" >/dev/null
  echo "Deleted: $id"
}

run_inference() {
  local prepared_json prepared_rows prepared_id models_json model_rows model_name destd frame_slice device batch_size
  local -a command
  prepared_json="$(run_cli list-prepared)"
  prepared_rows="$(choose_from_json_list "$prepared_json" prepared)"
  if ! prepared_id="$(select_row_id "Prepared input number" "$prepared_rows")"; then
    echo "Invalid selection."
    return
  fi

  models_json="$(run_cli list-models)"
  model_rows="$(JSON_PAYLOAD="$models_json" python - <<'PY'
import json, os
models = json.loads(os.environ["JSON_PAYLOAD"]).get("models", [])
for idx, model in enumerate(models, 1):
    print(f"{idx}|{model}|{model}|")
PY
)"
  if ! model_name="$(select_row_id "Model number" "$model_rows")"; then
    echo "Invalid selection."
    return
  fi
  destd="$(prompt "De-standardize predictions (true/false)" "true")"
  frame_slice="$(prompt "Frame slice start:stop:step (blank = all)" "")"
  device="$(prompt "Inference device" "cuda")"
  batch_size="$(prompt "Inference batch size" "1")"

  command=(infer-prepared "$prepared_id" --model "$model_name" --device "$device" --batch-size "$batch_size")
  if [ "$destd" = "true" ] || [ "$destd" = "True" ] || [ "$destd" = "1" ]; then
    command+=(--destandardize)
  else
    command+=(--no-destandardize)
  fi
  if [ -n "$frame_slice" ]; then
    command+=(--frame-slice "$frame_slice")
  fi
  if ! run_cli "${command[@]}"; then
    echo "Inference failed."
    return
  fi
}

while true; do
  echo ""
  echo "GEqNMR console"
  echo "Root: ${DATA_ROOT}"
  echo "1) List prepared inputs"
  echo "2) Add prepared input"
  echo "3) Run inference"
  echo "4) Delete prepared input"
  echo "5) List results"
  echo "6) Change root"
  echo "q) Quit"
  read -r -p "Action: " action
  case "$action" in
    1) list_prepared ;;
    2) add_prepared ;;
    3) run_inference ;;
    4) delete_prepared ;;
    5) list_results ;;
    6) change_root ;;
    q|Q|"") exit 0 ;;
    *) echo "Unknown action." ;;
  esac
done
