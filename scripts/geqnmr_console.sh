#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DATA_ROOT="${ROOT_DIR}/outputs"
API_URL="${GEQNMR_API_URL:-http://localhost:8000}"
DATA_ROOT="${GEQNMR_DATA_ROOT:-${DEFAULT_DATA_ROOT}}"

mkdir -p "${DATA_ROOT}/prepared_inputs"
chmod 755 "${DATA_ROOT}" "${DATA_ROOT}/prepared_inputs" 2>/dev/null || true
export GEQNMR_DATA_ROOT="${DATA_ROOT}"

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

api_get() {
  curl -fsS "$API_URL/$1"
}

api_delete() {
  curl -fsS -X DELETE "$API_URL/$1"
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
  local rows="$1"
  local label="$2"
  if [ -z "$rows" ]; then
    return 1
  fi
  print_rows "$rows"
  local choice
  choice="$(prompt "$label" "1")"
  awk -F'|' -v choice="$choice" '$1 == choice {print $2; found=1} END {exit found ? 0 : 1}' <<< "$rows"
}

list_prepared() {
  local json rows
  json="$(api_get prepared)"
  rows="$(choose_from_json_list "$json" prepared)"
  echo "Prepared inputs:"
  print_rows "$rows"
}

list_results() {
  local json
  json="$(api_get results)"
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
  local file traj name response job status
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

  if [ -n "$traj" ]; then
    response="$(curl -fsS -X POST "$API_URL/prepare" -F "file=@${file}" -F "trajectory_file=@${traj}" -F "name=${name}")"
  else
    response="$(curl -fsS -X POST "$API_URL/prepare" -F "file=@${file}" -F "name=${name}")"
  fi
  job="$(JSON_PAYLOAD="$response" python - <<'PY'
import json, os
print(json.loads(os.environ["JSON_PAYLOAD"])["job_id"])
PY
)"
  echo "Prepare job: $job"
  while true; do
    status="$(api_get "prepare/jobs/${job}")"
    JSON_PAYLOAD="$status" python - <<'PY'
import json, os
data = json.loads(os.environ["JSON_PAYLOAD"])
print(f"  {data.get('status')} {int(float(data.get('progress', 0))*100)}% - {data.get('message', '')}")
PY
    state="$(JSON_PAYLOAD="$status" python - <<'PY'
import json, os
print(json.loads(os.environ["JSON_PAYLOAD"]).get("status", ""))
PY
)"
    if [ "$state" = "completed" ]; then
      break
    fi
    if [ "$state" = "failed" ]; then
      return 1
    fi
    sleep 1
  done
}

delete_prepared() {
  local json rows id
  json="$(api_get prepared)"
  rows="$(choose_from_json_list "$json" prepared)"
  id="$(select_row_id "$rows" "Delete prepared input number")" || return
  api_delete "prepared/${id}" >/dev/null
  echo "Deleted: $id"
}

run_inference() {
  local prepared_json prepared_rows prepared_id models_json model_rows model_name destd frame_slice response
  prepared_json="$(api_get prepared)"
  prepared_rows="$(choose_from_json_list "$prepared_json" prepared)"
  prepared_id="$(select_row_id "$prepared_rows" "Prepared input number")" || return

  models_json="$(api_get models)"
  model_rows="$(JSON_PAYLOAD="$models_json" python - <<'PY'
import json, os
models = json.loads(os.environ["JSON_PAYLOAD"]).get("models", [])
for idx, model in enumerate(models, 1):
    print(f"{idx}|{model}|{model}|")
PY
)"
  model_name="$(select_row_id "$model_rows" "Model number")" || return
  destd="$(prompt "De-standardize predictions (true/false)" "true")"
  frame_slice="$(prompt "Frame slice start:stop:step (blank = all)" "")"

  if [ -n "$frame_slice" ]; then
    response="$(curl -fsS -X POST "$API_URL/infer/prepared/${prepared_id}" -F "model_name=${model_name}" -F "destandardize=${destd}" -F "frame_slice=${frame_slice}")"
  else
    response="$(curl -fsS -X POST "$API_URL/infer/prepared/${prepared_id}" -F "model_name=${model_name}" -F "destandardize=${destd}")"
  fi
  JSON_PAYLOAD="$response" python - <<'PY'
import json, os
data = json.loads(os.environ["JSON_PAYLOAD"])
print("Inference complete")
print(f"  output: {data.get('output_file')}")
print(f"  atoms:  {data.get('atoms_predicted')}")
PY
}

while true; do
  echo ""
  echo "GEqNMR console"
  echo "API:  ${API_URL}"
  echo "Root: ${DATA_ROOT}"
  echo "1) List prepared inputs"
  echo "2) Add prepared input"
  echo "3) Run inference"
  echo "4) Delete prepared input"
  echo "5) List results"
  echo "q) Quit"
  read -r -p "Action: " action
  case "$action" in
    1) list_prepared ;;
    2) add_prepared ;;
    3) run_inference ;;
    4) delete_prepared ;;
    5) list_results ;;
    q|Q|"") exit 0 ;;
    *) echo "Unknown action." ;;
  esac
done
