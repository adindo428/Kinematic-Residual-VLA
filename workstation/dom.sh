#!/usr/bin/env bash
set -euo pipefail
: "${KRVLA_REPO:?Source workstation/env.local.sh first}"
: "${POLICY_PYTHON:?Set POLICY_PYTHON}"
: "${STAGE:?Set STAGE}"; : "${ASSETS:?Set ASSETS}"; : "${RESULTS:?Set RESULTS}"
workstation_runner="$KRVLA_REPO/evaluation/official_dom/scripts/local.py"
workstation_action="${1:-help}"
if (($#)); then shift; fi
case "$workstation_action" in
  fetch) exec "$POLICY_PYTHON" "$workstation_runner" fetch --stage "$STAGE" "$@" ;;
  prepare) exec "$POLICY_PYTHON" "$workstation_runner" prepare --stage "$STAGE" --assets "$ASSETS" --root "$RESULTS" --gpu-label 'RTX PRO 6000 workstation' "$@" ;;
  pilot|full)
    : "${ISAACLAB:?Set ISAACLAB}"
    exec "$POLICY_PYTHON" "$workstation_runner" run --stage "$STAGE" --assets "$ASSETS" --root "$RESULTS" --isaaclab "$ISAACLAB" --policy-python "$POLICY_PYTHON" --mode "$workstation_action" "$@" ;;
  report) exec "$POLICY_PYTHON" "$workstation_runner" report --root "$RESULTS" "$@" ;;
  *) echo 'Usage: bash workstation/dom.sh {fetch|prepare|pilot|full|report} [extra arguments]'; exit 2 ;;
esac
