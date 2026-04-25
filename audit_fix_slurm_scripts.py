#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(r"c:\Users\Dipan\Desktop\prompt_gen\Stable_diffusion")

slurm_files = []
for f in ROOT.rglob("*.sh"):
    try:
        t = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if "#SBATCH" in t:
        slurm_files.append((f, t))

fixed = []
issues = []

for f, text in slurm_files:
    lines = text.splitlines()
    original = text

    # Ensure shebang exists at first line.
    if not lines or not lines[0].startswith("#!"):
        lines.insert(0, "#!/bin/bash")
        issues.append((str(f), "missing shebang (added)"))

    # Remove duplicate shebangs after first line.
    deduped = [lines[0]]
    for line in lines[1:]:
        if line.startswith("#!"):
            issues.append((str(f), "duplicate shebang (removed)"))
            continue
        deduped.append(line)
    lines = deduped

    # Remove any accidental literal backreference line.
    new_lines = []
    for line in lines:
        if line.strip() == r"\1":
            issues.append((str(f), "literal \\1 line (removed)"))
            continue
        new_lines.append(line)
    lines = new_lines

    uses_rdzv = any("--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" in l for l in lines)
    has_master_addr = any(l.strip().startswith("export MASTER_ADDR=") for l in lines)
    has_master_port = any(l.strip().startswith("export MASTER_PORT=") for l in lines)

    if uses_rdzv and not has_master_addr:
        # Insert before torchrun call if possible.
        insert_idx = 0
        for i, l in enumerate(lines):
            if "torchrun" in l:
                insert_idx = i
                break
        lines.insert(insert_idx, "export MASTER_ADDR=127.0.0.1")
        issues.append((str(f), "missing MASTER_ADDR (added)"))

    if uses_rdzv and not has_master_port:
        insert_idx = 0
        for i, l in enumerate(lines):
            if "torchrun" in l:
                insert_idx = i
                break
        lines.insert(insert_idx, "export MASTER_PORT=29500")
        issues.append((str(f), "missing MASTER_PORT (added)"))

    # Keep torchelastic capture for torchrun scripts.
    has_torchelastic = any(l.strip().startswith("export TORCHELASTIC_ERROR_FILE=") for l in lines)
    if uses_rdzv and not has_torchelastic:
        insert_idx = 0
        for i, l in enumerate(lines):
            if "torchrun" in l:
                insert_idx = i
                break
        lines.insert(insert_idx, "export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error_${SLURM_JOB_ID}.json")
        issues.append((str(f), "missing TORCHELASTIC_ERROR_FILE (added)"))

    new_text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    if new_text != original:
        f.write_text(new_text, encoding="utf-8")
        fixed.append(str(f))

print(f"Slurm scripts found: {len(slurm_files)}")
print(f"Files modified: {len(fixed)}")
if fixed:
    print("Modified files:")
    for p in fixed:
        print(f" - {p}")

if issues:
    print("Issues handled:")
    for p, msg in issues:
        print(f" - {p}: {msg}")
