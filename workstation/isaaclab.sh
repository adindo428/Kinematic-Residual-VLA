#!/usr/bin/env bash
# Explicitly selects the simulator environment, even if the policy is active.
set -euo pipefail
: "${ISAACLAB_ROOT:?Source workstation/env.local.sh first}"
: "${ISAACLAB_ENV:?Set the simulator conda environment name}"
exec conda run --no-capture-output -n "$ISAACLAB_ENV" bash "$ISAACLAB_ROOT/isaaclab.sh" "$@"
