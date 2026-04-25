#!/usr/bin/env python3
from pathlib import Path

base = Path(r"c:\Users\Dipan\Desktop\prompt_gen\Stable_diffusion")
updated = 0

for f in base.rglob("*.sh"):
    try:
        text = f.read_text(encoding="utf-8", errors="ignore")
        original = text

        # Repair accidental literal backreference line from previous script.
        text = text.replace("\n\\1\n", "\nexport MASTER_PORT=29500\n")

        # If still missing, insert MASTER_PORT after MASTER_ADDR export.
        if "export MASTER_PORT=" not in text and "export MASTER_ADDR=" in text:
            lines = text.splitlines()
            out = []
            inserted = False
            for line in lines:
                out.append(line)
                if (not inserted) and line.strip().startswith("export MASTER_ADDR="):
                    out.append("export MASTER_PORT=29500")
                    inserted = True
            text = "\n".join(out) + ("\n" if text.endswith("\n") else "")

        if text != original:
            f.write_text(text, encoding="utf-8")
            print(f"Updated: {f}")
            updated += 1
    except Exception as e:
        print(f"Error: {f} -> {e}")

print(f"Total updated: {updated}")
