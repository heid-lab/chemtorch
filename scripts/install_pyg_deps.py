#!/usr/bin/env python3
"""
Detect the active environment's PyTorch version and build CUDA tag,
print the recommended pip command to install PyG wheel dependencies, and
optionally run it.

Usage:
  python scripts/install_pyg_deps.py        # dry-run, prints the pip command
  python scripts/install_pyg_deps.py --run  # actually runs the pip install
  python scripts/install_pyg_deps.py -y     # same as --run
"""
from __future__ import annotations
import argparse
import re
import shlex
import subprocess
import sys

PKGS = "pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv"

def detect_torch() -> tuple[str, str]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Failed to import torch in the active environment.") from exc

    # torch.__version__ might be "2.2.0+cu118" or "2.2.0" or "2.2.0.dev...".
    raw_version = getattr(torch, "__version__", None)
    if not raw_version:
        raise RuntimeError("torch.__version__ is not available.")

    base_version = re.split(r"[+\-]", raw_version)[0]

    # torch.version.cuda is the build-time cuda string (e.g. "11.8", "12.4") or None
    build_cuda = getattr(torch.version, "cuda", None)
    if build_cuda is None:
        cuda_tag = "cpu"
    else:
        # convert "11.8" -> "cu118", "12.4" -> "cu124", "13.0" -> "cu130"
        parts = re.findall(r"\d+", build_cuda)
        if not parts:
            cuda_tag = "cpu"
        else:
            cuda_tag = "cu" + "".join(parts)

    return base_version, cuda_tag

def make_pip_cmd(torch_ver: str, cuda_tag: str) -> str:
    wheel_index = f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"
    return f"pip install {PKGS} -f {shlex.quote(wheel_index)}"

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Install PyTorch Geometric binary deps for the current torch build")
    ap.add_argument("--run", "-y", action="store_true", help="actually run the pip install (default: show command only)")
    args = ap.parse_args(argv)

    try:
        torch_ver, cuda_tag = detect_torch()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        print("Activate the Python environment that has torch installed and re-run this script.", file=sys.stderr)
        return 2

    pip_cmd = make_pip_cmd(torch_ver, cuda_tag)

    print("Detected PyTorch build:")
    print(f"  torch=={torch_ver}")
    print(f"  PyTorch build CUDA tag: {cuda_tag}")
    print()
    print("Recommended command to install PyG optional wheels:")
    print()
    print(f"  {pip_cmd}")
    print()

    if args.run:
        print("Running:", pip_cmd)
        try:
            # Use shell=False with simple tokenization to be safer across platforms
            # We purposely call pip as a subprocess which will use the active environment's pip.
            tokens = ["pip"] + pip_cmd.split()[2:]  # drop the leading 'pip' then reuse args
            subprocess.check_call(tokens)
            print("PyG wheel installation finished.")
        except subprocess.CalledProcessError as e:
            print("pip install failed with exit code", e.returncode, file=sys.stderr)
            return e.returncode
    else:
        print("Dry-run: re-run with --run or -y to execute the command.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())