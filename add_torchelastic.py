#!/usr/bin/env python3
import re
from pathlib import Path
base_dir = Path(r"c:\Users\Dipan\Desktop\prompt_gen\Stable_diffusion")
pattern = r'(export MASTER_PORT=29500)'
replacement = r"\\1\nexport TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error_${SLURM_JOB_ID}.json"
count = 0
for f in base_dir.rglob('*.sh'):
    try:
        s = f.read_text(encoding='utf-8', errors='ignore')
        if 'TORCHELASTIC_ERROR_FILE' in s:
            continue
        if 'export MASTER_PORT=29500' in s:
            new = re.sub(pattern, replacement, s)
            if new != s:
                f.write_text(new, encoding='utf-8')
                print('Updated:', f)
                count += 1
    except Exception as e:
        print('Error', f, e)
print('Total updated:', count)
