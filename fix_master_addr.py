#!/usr/bin/env python3
import os
import re
from pathlib import Path

# Directory to search
base_dir = Path(r"c:\Users\Dipan\Desktop\prompt_gen\Stable_diffusion")

# Pattern to find
old_pattern = r'export MASTER_ADDR=\$\(scontrol show hostnames "\$SLURM_JOB_NODELIST" \| head -n 1\)-hsn0'
new_replacement = 'export MASTER_ADDR=127.0.0.1'

# Alternative pattern (with quotes)
old_pattern2 = r'MASTER_ADDR="\$\(scontrol show hostnames "\$SLURM_JOB_NODELIST" \| head -n 1\)-hsn0"'
new_replacement2 = 'MASTER_ADDR="127.0.0.1"'

fixed_count = 0

for sh_file in base_dir.rglob("*.sh"):
    try:
        with open(sh_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # Replace both patterns
        content = re.sub(old_pattern, new_replacement, content)
        content = re.sub(old_pattern2, new_replacement2, content)
        
        if content != original_content:
            with open(sh_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {sh_file}")
            fixed_count += 1
    except Exception as e:
        print(f"Error processing {sh_file}: {e}")

print(f"\nTotal files fixed: {fixed_count}")
