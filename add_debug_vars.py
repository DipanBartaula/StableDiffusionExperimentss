#!/usr/bin/env python3
import os
import re
from pathlib import Path

base_dir = Path(r"c:\Users\Dipan\Desktop\prompt_gen\Stable_diffusion")

# Pattern to find MASTER_PORT line
pattern = r'(export MASTER_PORT=29500)'

# Add debug variables after MASTER_PORT
replacement = r'\1\nexport NCCL_DEBUG=INFO\nexport TORCH_DISTRIBUTED_DEBUG=DETAIL'

fixed_count = 0

for sh_file in base_dir.rglob("*.sh"):
    try:
        with open(sh_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original_content = content
        
        # Only add if not already present and has MASTER_PORT
        if 'NCCL_DEBUG' not in content and 'export MASTER_PORT=29500' in content:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(sh_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {sh_file}")
            fixed_count += 1
    except Exception as e:
        print(f"Error processing {sh_file}: {e}")

print(f"\nTotal files updated with debug variables: {fixed_count}")
