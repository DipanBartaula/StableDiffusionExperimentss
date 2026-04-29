#!/usr/bin/env python3
from pathlib import Path
import json
import re
import subprocess
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / 'evaluate.py'
CKPT_DIR = Path('/iopsstor/scratch/cscs/dbartaula/experiments_assets_1/Stable_diffusion_train_medium_hard_soft_curriculum_catvton/checkpoints')
RUN_NAME = CKPT_DIR.parent.name
COMMAND = ['python', str(EVAL_SCRIPT), '--checkpoint', '__CKPT_PATH__', '--curvton_test_data_path', '/iopsstor/scratch/cscs/dbartaula/human_gen/dataset_v3_backup_1/dataset_ultimate_test', '--triplet_test_data_path', '', '--street_tryon_data_path', '', '--batch_size', '16', '--num_workers', '8', '--eval_frac_curvton', '0.25', '--curvton_splits', 'medium,hard']


def _latest_ckpt(ckpt_dir: Path) -> Path:
    final_ckpt = ckpt_dir / 'ckpt_final.pt'
    if final_ckpt.exists():
        return final_ckpt

    def step_num(p: Path) -> int:
        m = re.search(r'ckpt_step_(\d+)\.pt$', p.name)
        return int(m.group(1)) if m else -1

    candidates = sorted(ckpt_dir.glob('ckpt_step_*.pt'), key=step_num)
    if not candidates:
        raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')
    return candidates[-1]




def _print_output_json_if_available(cmd_tokens):
    if '--output_json' not in cmd_tokens:
        return
    try:
        idx = cmd_tokens.index('--output_json')
        out_path = Path(cmd_tokens[idx + 1])
    except (ValueError, IndexError):
        return

    if not out_path.exists():
        return

    try:
        data = json.loads(out_path.read_text(encoding='utf-8'))
        print('\nMetrics summary (from output_json):')
        print(json.dumps(data, indent=2))
    except Exception as exc:
        print(f'WARNING: could not read output JSON {out_path}: {exc}')

def main() -> int:
    ckpt_path = None
    if CKPT_DIR.exists():
        try:
            ckpt_path = _latest_ckpt(CKPT_DIR)
        except FileNotFoundError:
            ckpt_path = None

    if ckpt_path is not None:
        cmd = [str(ckpt_path) if t == '__CKPT_PATH__' else t for t in COMMAND]
    else:
        cmd = []
        skip_next = False
        for i, t in enumerate(COMMAND):
            if skip_next:
                skip_next = False
                continue
            if t == '--checkpoint' and i + 1 < len(COMMAND) and COMMAND[i + 1] == '__CKPT_PATH__':
                skip_next = True
                continue
            cmd.append(t)

    print(f'Evaluating run: {RUN_NAME}')
    if ckpt_path is not None:
        print(f'Using checkpoint: {ckpt_path}')
    else:
        print(f'No checkpoint available in: {CKPT_DIR}')
        print('Evaluating with initial (Xavier/init) weights.')
    print('Command:', ' '.join(cmd))

    extra_args = sys.argv[1:]
    if extra_args:
        print('Forwarding extra CLI args:', ' '.join(extra_args))
        cmd = cmd + extra_args

    result = subprocess.run(cmd)
    if result.returncode == 0:
        _print_output_json_if_available(cmd)
    return result.returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)








