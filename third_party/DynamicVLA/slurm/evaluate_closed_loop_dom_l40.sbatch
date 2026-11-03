#!/usr/bin/env bash
#SBATCH --job-name=dvla-dom-closed-loop
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=l40:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.out
#SBATCH --error=/home/aparame/Research/Dynamic_VLA/logs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=/home/aparame/Research/Dynamic_VLA/third_party/DynamicVLA
SIF_FILE=${SIF_FILE:-/scratch/aparame/isaaclab.sif}
ASSET_ROOT=${ASSET_ROOT:-/scratch/aparame/Dynamic_VLA/dom_benchmark}
SCENE_DIR=${SCENE_DIR:-${ASSET_ROOT}/scenes}
OBJECT_DIR=${OBJECT_DIR:-${ASSET_ROOT}/objects}
ENV_CFG=${ENV_CFG:-${ASSET_ROOT}/tests/1-1_place_franka_apple13d_O02_01048988_09b0.json}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/scratch/aparame/Dynamic_VLA/runs/checkpoints}
OUTPUT_ROOT=${OUTPUT_ROOT:-/scratch/aparame/Dynamic_VLA/runs/closed_loop/${SLURM_JOB_ID}}
LOG_ROOT=/home/aparame/Research/Dynamic_VLA/logs/closed_loop/${SLURM_JOB_ID}
ISAAC_SIM_DATA=${ISAAC_SIM_DATA:-/scratch/aparame/isaac-sim}
HF_HOME=${HF_HOME:-/scratch/aparame/.cache/huggingface}
ISAAC_DEPS=${ISAAC_DEPS:-/scratch/aparame/Dynamic_VLA/isaac_eval_deps/python}
N_TESTS=${N_TESTS:-1}
EVAL_VARIANTS=${EVAL_VARIANTS:-baseline residual}
IMG_PORT=${IMG_PORT:-3186}
ACT_PORT=${ACT_PORT:-3188}

for required in "${SIF_FILE}" "${SCENE_DIR}" "${OBJECT_DIR}" "${ENV_CFG}" "${ISAAC_DEPS}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing DOM benchmark dependency: ${required}" >&2
        exit 2
    fi
done
if [[ "${ENV_CFG}" == *.txt ]] && [[ ! -s "${ENV_CFG}" ]]; then
    echo "DOM test list is empty: ${ENV_CFG}" >&2
    exit 2
fi
if ! find "${SCENE_DIR}" -type f -name '*.usd' -print -quit | grep -q .; then
    echo "No DOM scene USD files found in ${SCENE_DIR}" >&2
    exit 2
fi
if ! find "${OBJECT_DIR}" -type f -name '*.usd' -print -quit | grep -q .; then
    echo "No DOM object USD files found in ${OBJECT_DIR}" >&2
    exit 2
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is unavailable on this node." >&2
    exit 2
fi
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}" "${ISAAC_SIM_DATA}/kit_data" \
    "${ISAAC_SIM_DATA}/cache/kit" "${ISAAC_SIM_DATA}/tmp"
chmod 1777 "${ISAAC_SIM_DATA}/tmp"

source "${PROJECT_DIR}/scripts/palmetto_env.sh"
cd "${ASSET_ROOT}"
export TERM=xterm
server_pid=
cleanup() {
    if [[ -n "${server_pid}" ]]; then
        kill -- -"${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for variant in ${EVAL_VARIANTS}; do
    if [[ "${variant}" != baseline && "${variant}" != residual ]]; then
        echo "Unsupported EVAL_VARIANTS entry: ${variant}" >&2
        exit 2
    fi
    checkpoint="${CHECKPOINT_ROOT}/$(
        case "${variant}" in
            baseline) echo dynamicvla_dom_full_finetune_speed_ood_0p10_0p18 ;;
            residual) echo residual_dynamicvla_dom_full_finetune_speed_ood_0p10_0p18 ;;
        esac
    )"
    for required in "${checkpoint}/config.json" "${checkpoint}/model.safetensors"; do
        if [[ ! -f "${required}" ]]; then
            echo "Missing checkpoint file: ${required}" >&2
            exit 2
        fi
    done
    echo "Evaluating ${variant} with ${checkpoint}"
    setsid apptainer exec --cleanenv --nv \
        -B "${PROJECT_DIR}:${PROJECT_DIR}" \
        -B "${ASSET_ROOT}:${ASSET_ROOT}" \
        -B "${ISAAC_DEPS}:${ISAAC_DEPS}" \
        -B "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
        -B "${ISAAC_SIM_DATA}/kit_data:/isaac-sim/kit/data" \
        -B "${ISAAC_SIM_DATA}/cache/kit:/isaac-sim/kit/cache" \
        -B "${ISAAC_SIM_DATA}/tmp:/tmp" \
        --env "PYTHONPATH=${ISAAC_DEPS}:${PROJECT_DIR}" \
        --env "OMNI_KIT_ACCEPT_EULA=YES" \
        --env "TERM=xterm" \
        --pwd "${ASSET_ROOT}" \
        "${SIF_FILE}" /opt/IsaacLab/isaaclab.sh -p \
        "${PROJECT_DIR}/simulations/evaluate.py" --headless --enable_cameras \
        --device cuda:0 --num_envs 1 --scene_dir "${SCENE_DIR}" \
        --object_dir "${OBJECT_DIR}" --env_cfg "${ENV_CFG}" \
        --output_dir "${OUTPUT_ROOT}/${variant}" -n "${N_TESTS}" \
        --img_port "${IMG_PORT}" --act_port "${ACT_PORT}" \
        >"${LOG_ROOT}/${variant}.server.log" 2>&1 &
    server_pid=$!
    if ! timeout 45m python "${PROJECT_DIR}/scripts/inference.py" \
        -p "${checkpoint}" -a "${variant}" -r euler -d \
        --img_port "${IMG_PORT}" --act_port "${ACT_PORT}" \
        -o "${OUTPUT_ROOT}/${variant}/inference" \
        >"${LOG_ROOT}/${variant}.client.log" 2>&1; then
        echo "${variant} policy inference failed; see ${LOG_ROOT}" >&2
        exit 1
    fi
    kill -- -"${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    server_pid=
done
