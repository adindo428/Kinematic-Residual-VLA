#!/usr/bin/env bash
# Shared runner; individual production sbatch files remain independent.
set -euo pipefail
source scripts/dom100_env.sh
command_name=$1
shift
case "${command_name}" in
    prepare)
        selection_args=()
        if [[ "${DOM100_AVAILABLE_ONLY:-0}" == 1 ]]; then
            selection_args+=(--available-only)
        elif [[ -n "${DOM100_PREFETCH:-}" ]]; then
            selection_args+=(--prefetch)
        fi
        exec python scripts/prepare_dom100.py --family-root "${DOM100_ROOT}" \
            --dataset-root "${HF_LEROBOT_HOME}/hzxie/DOM" \
            --model-revision "${DOM100_MODEL_REVISION:-main}" \
            --dataset-revision "${DOM100_DATASET_REVISION:-main}" "${selection_args[@]}"
        ;;
    profile)
        precision_args=()
        if [[ -n "${DOM100_PRECISION:-}" ]]; then precision_args+=(--precision "${DOM100_PRECISION}"); fi
        exec python scripts/dom100.py profile --family-root "${DOM100_ROOT}" \
            --output "${DOM100_ROOT}/profiling/${SLURM_JOB_ID:-local}" "${precision_args[@]}"
        ;;
    integration)
        bind_args=()
        if [[ -n "${DOM100_CASE_MAP:-}" ]]; then bind_args+=(--case-map "${DOM100_CASE_MAP}"); fi
        python scripts/dom_closed_loop.py bind --family-root "${DOM100_ROOT}" "${bind_args[@]}"
        exec python scripts/dom_closed_loop.py integration --family-root "${DOM100_ROOT}"
        ;;
    smoke) exec python scripts/dom100.py smoke --family-root "${DOM100_ROOT}" ;;
    train|pilots) ;;
    *) echo "Unknown DOM100 command: ${command_name}" >&2; exit 2 ;;
esac
args=("${command_name}" --family-root "${DOM100_ROOT}")
if [[ "${command_name}" == train ]]; then args+=(--variant "$1"); fi
child=
forward_stop() { if [[ -n "${child}" ]]; then kill -USR1 "${child}" 2>/dev/null || true; fi; }
trap forward_stop USR1 TERM
python scripts/dom100.py "${args[@]}" &
child=$!
set +e
wait "${child}"
status=$?
# A trapped shell signal interrupts wait before the child finishes saving.
while kill -0 "${child}" 2>/dev/null; do wait "${child}"; status=$?; done
if (( status > 128 )); then wait "${child}"; status=$?; fi
set -e
if [[ ${status} -eq 75 && -n "${SLURM_JOB_ID:-}" ]]; then
    if scontrol requeue "${SLURM_JOB_ID}"; then exit 0; fi
    echo "Requeue unavailable. Resubmit the same sbatch file to resume the same run." >&2
fi
exit "${status}"
