# src/axon.py

from __future__ import annotations
from typing import Optional, Dict, Any
import os
import subprocess
import shutil

def run_axondeepseg_cli(input_path: str, output_dir: str, model: str = "default") -> Dict[str, Any]:
    """
    Try to call the AxonDeepSeg CLI if installed.
    """
    ads = shutil.which("AxonDeepSeg")
    if ads is None:
        raise RuntimeError(
            "AxonDeepSeg CLI not found. Install 'AxonDeepSeg' to enable optic nerve analysis."
        )
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        ads,
        "-i", input_path,
        "-o", output_dir,
        "-t", model
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"AxonDeepSeg failed: {result.stderr}")
    return {"ok": True, "stdout": result.stdout}

def analyze_optic_nerve(input_dir: str, output_dir: str, model: str = "default") -> Dict[str, Any]:
    """
    Process all images in input_dir with AxonDeepSeg if available.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Optic nerve input directory not found: {input_dir}")
    results = {}
    for name in os.listdir(input_dir):
        if not name.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            continue
        inp = os.path.join(input_dir, name)
        outd = os.path.join(output_dir, os.path.splitext(name)[0])
        try:
            r = run_axondeepseg_cli(inp, outd, model=model)
            results[name] = r
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
    return results

