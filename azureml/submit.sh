#!/usr/bin/env bash
# Submit a picotron job to the ScalingExp cluster.
#
# WANDB_API_KEY (needed for --use_wandb) is read from, in order of precedence:
#   1. the environment, if already exported
#   2. a .env file: $WANDB_ENV_FILE, else ./.env, else ../.env (repo root)
# Never hardcode the key in this file.
#
#   ./submit.sh step6                   # step1 | step3 | step5 | step6
#
# step5/step6 are the real data-parallel baselines. step3 has no gradient
# all-reduce; step1 is a single-step smoke test on random data.
#
# Requires: az login, and the workspace/resource-group defaults set below.
set -euo pipefail

STEP="${1:-step6}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_YML="$HERE/job-${STEP}.yml"

# Verified against `az ml compute show --name ScalingExp`.
RESOURCE_GROUP="${RESOURCE_GROUP:-defaultresourcegroup-northeurope}"
WORKSPACE="${WORKSPACE:-ProductPlacementDev}"

if [[ ! -f "$JOB_YML" ]]; then
  echo "No such job file: $JOB_YML" >&2
  echo "Available:" >&2
  ls "$HERE"/job-*.yml >&2
  exit 1
fi

# Load WANDB_API_KEY from a .env file only if it isn't already set, so an
# explicit `export WANDB_API_KEY=...` still takes precedence. We read just that
# one key rather than `source`-ing the file, so arbitrary shell in .env can't run.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  for envfile in "${WANDB_ENV_FILE:-}" "$HERE/.env" "$HERE/../.env"; do
    [[ -n "$envfile" && -f "$envfile" ]] || continue
    line="$(grep -E '^[[:space:]]*(export[[:space:]]+)?WANDB_API_KEY=' "$envfile" | tail -n1)"
    if [[ -n "$line" ]]; then
      val="${line#*=}"                       # strip up to first '='
      val="${val%%$'\r'}"                     # drop any trailing CR (CRLF files)
      val="${val#[\"\']}"; val="${val%[\"\']}" # strip one layer of surrounding quotes
      export WANDB_API_KEY="$val"
      echo "Loaded WANDB_API_KEY from $envfile"
      break
    fi
  done
fi

# This box has two Azure CLIs and only one has a working `ml` extension:
# /opt/az's copy dies on `No module named 'rpds.rpds'`, and because the import
# fails the CLI decides `ml` is missing and tries to install it into root-owned
# /opt/az, which then explodes with permission errors. So don't trust PATH --
# probe for an az whose `ml` extension actually loads. Override with AZ_BIN.
AZ_BIN="${AZ_BIN:-}"
if [[ -z "$AZ_BIN" ]]; then
  for cand in \
      /anaconda/envs/azureml_py38/bin/az \
      "$(command -v az 2>/dev/null || true)" \
      /opt/az/bin/az; do
    [[ -n "$cand" && -x "$cand" ]] || continue
    if "$cand" ml -h >/dev/null 2>&1; then
      AZ_BIN="$cand"
      break
    fi
  done
fi

if [[ -z "$AZ_BIN" ]]; then
  echo "No Azure CLI with a working 'ml' extension found." >&2
  echo "Set AZ_BIN=/path/to/az explicitly, or repair the extension with:" >&2
  echo "  az extension remove -n ml && az extension add -n ml" >&2
  exit 1
fi
echo "Using az: $AZ_BIN"

if ! "$AZ_BIN" account show >/dev/null 2>&1; then
  echo "Not logged in for this CLI. Run: $AZ_BIN login" >&2
  exit 1
fi

ARGS=(--file "$JOB_YML" --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE")

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  ARGS+=(--set "environment_variables.WANDB_API_KEY=$WANDB_API_KEY")
elif grep -q '^  --use_wandb$' "$JOB_YML"; then
  # The run would start, train, then die at wandb.init() -- fail fast instead.
  echo "WANDB_API_KEY is unset but $JOB_YML passes --use_wandb." >&2
  echo "Export the key, or drop --use_wandb from the job's command." >&2
  exit 1
fi

"$AZ_BIN" ml job create "${ARGS[@]}" --web
