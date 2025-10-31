#!/usr/bin/env python3
"""Collect hardware and software environment info and dump as YAML.

Usage:
  python scripts/collect_env.py --out test/test_integration/fixtures/my_models/baseline_env.yaml

This helper is intended to be run on the machine where you generate reference
predictions. It captures CPU/GPU details, important package versions, and a
short pip freeze snapshot.
"""
from __future__ import annotations
import argparse
import platform
import subprocess
import yaml
import sys


def run_cmd(cmd: str):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception as e:
        return None


def collect():
    info = {}
    info["host"] = {
        "uname": platform.uname()._asdict(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    # CPU
    cpu = {}
    cpu["lscpu"] = run_cmd("lscpu")
    cpu["nproc"] = run_cmd("nproc --all")
    info["cpu"] = cpu

    # GPU (NVIDIA)
    gpu = {}
    gpu["nvidia_smi"] = run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    gpu["lspci"] = run_cmd("lspci | grep -i -E 'nvidia|amd|vga|3d' || true")
    info["gpu"] = gpu

    # Packages of interest
    pkgs = {}
    for pkg in ("torch", "numpy", "pandas", "hydra", "omegaconf"):
        try:
            m = __import__(pkg)
            ver = getattr(m, "__version__", getattr(m, "version", "unknown"))
            pkgs[pkg] = {"version": str(ver)}
            if pkg == "torch":
                try:
                    pkgs[pkg]["cuda_version"] = m.version.cuda if hasattr(m, "version") else None
                    pkgs[pkg]["cudnn_version"] = m.backends.cudnn.version() if hasattr(m.backends, "cudnn") else None
                    if m.cuda.is_available():
                        pkgs[pkg]["cuda_devices"] = [m.cuda.get_device_name(i) for i in range(m.cuda.device_count())]
                except Exception:
                    pass
        except Exception as e:
            pkgs[pkg] = {"installed": False, "error": str(e)}
    info["packages"] = pkgs

    # numpy config (BLAS info)
    try:
        import numpy as np
        import io
        buf = io.StringIO()
        np.__config__.show(log=buf)
        info["numpy_config"] = buf.getvalue()
    except Exception:
        info["numpy_config"] = None

    # pip freeze (short)
    info["pip_freeze"] = run_cmd("pip freeze --all | sed -n '1,200p'")

    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", help="Write YAML to path", default=None)
    args = p.parse_args()

    info = collect()
    yaml_out = yaml.safe_dump(info, sort_keys=False)
    sys.stdout.write(yaml_out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(yaml_out)
        print("Wrote", args.out)


if __name__ == "__main__":
    main()
