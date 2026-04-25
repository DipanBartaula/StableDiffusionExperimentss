from pathlib import Path
import re

root = Path(r"c:\Users\Dipan\Desktop\prompt_gen\Stable_diffusion")

files = []
for p in root.rglob("*.sh"):
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if "srun torchrun" in t:
        files.append(p)

updated = []

for p in files:
    text = p.read_text(encoding="utf-8", errors="ignore")
    original = text

    # Ensure strict bash mode exists.
    if "set -euo pipefail" not in text:
        text = text.replace("#SBATCH --error=logs/%x_%j.err\n", "#SBATCH --error=logs/%x_%j.err\n\nset -euo pipefail\n", 1)

    # Replace naive PYTHONPATH export with robust conda activation block if not already present.
    old_py = 'cd "$WORK_DIR"\nexport PYTHONPATH="$WORK_DIR:$PYTHONPATH"\n'
    new_py = (
        'cd "$WORK_DIR"\n\n'
        '# Clean inherited Python env vars first; bad PYTHONHOME/PYTHONPATH can break stdlib encodings.\n'
        'unset PYTHONHOME || true\n'
        'unset PYTHONPATH || true\n\n'
        '# Activate the intended conda env used on the cluster.\n'
        'CONDA_ROOT="/iopsstor/scratch/cscs/dbartaula/miniforge3"\n'
        'CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch26_env_new}"\n'
        'if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then\n'
        '  source "$CONDA_ROOT/etc/profile.d/conda.sh"\n'
        'else\n'
        '  source "$HOME/miniforge3/etc/profile.d/conda.sh"\n'
        'fi\n'
        'conda activate "$CONDA_ENV_NAME"\n\n'
        'export PYTHONNOUSERSITE=1\n'
        'export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"\n'
    )
    if old_py in text and "CONDA_ENV_NAME" not in text:
        text = text.replace(old_py, new_py)

    # Normalize old safe PYTHONPATH line if present and conda block exists.
    if "CONDA_ENV_NAME" in text:
        text = text.replace('export PYTHONPATH="$WORK_DIR:$PYTHONPATH"', 'export PYTHONPATH="$WORK_DIR:${PYTHONPATH:-}"')

    # Replace network env block with interface autodetect block.
    net_pat = re.compile(
        r'export NCCL_SOCKET_IFNAME=.*?\n'
        r'export NCCL_NET_GDR_LEVEL=PHB\n'
        r'export NCCL_CROSS_NIC=1\n'
        r'export FI_CXI_ATS=0\n'
        r'export GLOO_SOCKET_IFNAME=.*?\n',
        re.DOTALL,
    )
    net_block = (
        'export NCCL_SOCKET_IFNAME=hsn\n'
        'export NCCL_NET_GDR_LEVEL=PHB\n'
        'export NCCL_CROSS_NIC=1\n'
        'export FI_CXI_ATS=0\n'
        'export GLOO_SOCKET_IFNAME=hsn\n\n'
        '# Select a concrete network interface name for Gloo/NCCL ("hsn" alone is not a valid device).\n'
        'for _if in hsn0 ib0 eth0 enp0s3 lo; do\n'
        '  if ip -o link show "$_if" >/dev/null 2>&1; then\n'
        '    export NCCL_SOCKET_IFNAME="$_if"\n'
        '    export GLOO_SOCKET_IFNAME="$_if"\n'
        '    break\n'
        '  fi\n'
        'done\n'
    )
    text = re.sub(net_pat, net_block, text, count=1)

    # Ensure debugging env vars are present.
    if "export DISABLE_WANDB=1" not in text and "export TORCH_DISTRIBUTED_DEBUG=DETAIL" in text:
        text = text.replace(
            "export TORCH_DISTRIBUTED_DEBUG=DETAIL\n",
            "export TORCH_DISTRIBUTED_DEBUG=DETAIL\nexport DISABLE_WANDB=1\nexport WANDB_MODE=disabled\nexport WANDB_SILENT=true\n",
            1,
        )

    # Convert rendezvous args to standalone for single-node scripts.
    text = text.replace("  --rdzv_backend=c10d \\\n", "")
    text = text.replace("  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \\\n", "")
    text = text.replace("  --rdzv_id=$SLURM_JOB_ID \\\n", "")
    if "--standalone" not in text:
        text = text.replace("  --nproc_per_node=4 \\\n", "  --nproc_per_node=4 \\\n  --standalone \\\n  --master_port=$MASTER_PORT \\\n", 1)

    # Ensure a useful debug line exists.
    if "[DEBUG] NCCL_SOCKET_IFNAME" not in text and "[DEBUG] Host=" in text:
        text = text.replace(
            'echo "[DEBUG] Host=$(hostname) JobID=${SLURM_JOB_ID:-unknown}"\n',
            'echo "[DEBUG] Host=$(hostname) JobID=${SLURM_JOB_ID:-unknown}"\n'
            'echo "[DEBUG] Conda env=$CONDA_ENV_NAME CONDA_PREFIX=${CONDA_PREFIX:-unset}"\n'
            'echo "[DEBUG] NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"\n',
            1,
        )

    if text != original:
        p.write_text(text, encoding="utf-8")
        updated.append(p)

print(f"Torchrun scripts found: {len(files)}")
print(f"Updated: {len(updated)}")
for p in updated:
    print(p)
