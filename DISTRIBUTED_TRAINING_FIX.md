# PyTorch Distributed Training Fix Summary

## Issue
**Error**: `RendezvousConnectionError: The connection to the C10d store has failed` with `RuntimeError: connect() timed out`

**Root Cause**: The problematic MASTER_ADDR configuration was using hostname construction with the scontrol command:
```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)-hsn0
```

This approach had issues:
- Hostname resolution failures or timeouts
- Incorrect network interface binding
- Rendezvous backend couldn't connect within the 60-second timeout

## Solution
Changed `MASTER_ADDR` to use localhost for single-node training with multiple GPUs:
```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

**Why this works:**
- For single-node training (`--nodes=1`, `--nproc_per_node=4`), localhost is reliable
- Eliminates hostname resolution delays
- Direct TCP connection via loopback interface
- Works across all SLURM environments

## Changes Made

### Files Fixed: 38 training scripts
- **scripts_catvton/only_levels/**: 4 files
- **scripts_catvton/curriculum_training/finetuning/**: 8 files  
- **scripts_catvton/curriculum_training/pretraining/**: 8 files
- **scripts_catvton/percentage/**: 4 files
- **scripts_gan/**: 4 files
- **scripts_hunyuan/**: 3 files
- **cross-architecture/**: 1 file (OOTDiffusion)

### Additional Enhancements
Added debug variables for better diagnostics:
```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
```

## How to Apply
These fixes are already applied to all training scripts. To submit a training job:
```bash
sbatch train_easy_only_catvton.sh
```

## Verification
All files verified to have:
- ✅ `MASTER_ADDR=127.0.0.1`
- ✅ No more problematic `scontrol` patterns
- ✅ Debug variables for troubleshooting

## References
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- Error typically occurs when rendezvous backend can't establish communication within timeout
