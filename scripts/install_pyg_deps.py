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
import subprocess
import sys
import shlex

PKGS = "pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv"
DOC_URL = "https://heid-lab.github.io/chemtorch/getting_started/quick_start.html#pyg-installation"

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
        backend_tag = "cpu"
    else:
        # convert "11.8" -> "cu118", "12.4" -> "cu124", "13.0" -> "cu130"
        parts = re.findall(r"\d+", build_cuda)
        if not parts:
            backend_tag = "cpu"
        else:
            backend_tag = "cu" + "".join(parts)

    return base_version, backend_tag

def normalize_torch_version(torch_ver: str) -> str:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?$", torch_ver)
    if not match:
        return torch_ver
    major, minor, _patch = match.groups()
    return f"{major}.{minor}.0"

def make_pip_cmd(torch_ver: str, backend_tag: str) -> tuple[str, str]:
    wheel_version = normalize_torch_version(torch_ver)
    wheel_index = f"https://data.pyg.org/whl/torch-{wheel_version}+{backend_tag}.html"
    return f"uv pip install {PKGS} -f {shlex.quote(wheel_index)}", wheel_version

def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(description="Install PyTorch Geometric binary deps for the current torch build")
    ap.add_argument("--run", "-y", action="store_true", help="actually run the pip install (default: show command only)")
    args = ap.parse_args(argv)

    try:
        torch_ver, backend_tag = detect_torch()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        print("Activate the Python environment that has torch installed and re-run this script.", file=sys.stderr)
        return 2

    pip_cmd, wheel_version = make_pip_cmd(torch_ver, backend_tag)

    print("Detected PyTorch build:")
    print(f"  torch=={torch_ver}")
    print(f"  PyTorch backend: {backend_tag}")
    if wheel_version != torch_ver:
        print(f"  PyG wheel lookup: torch=={wheel_version}")
    print()
    print("Recommended command to install PyG optional wheels:")
    print()
    print(f"  {pip_cmd}")
    print()

    if args.run:
        print("Running:", pip_cmd)
        try:
            # Tokenize the command safely so quoted URLs remain intact.
            tokens = shlex.split(pip_cmd)
            subprocess.check_call(tokens)
            print("PyG wheel installation finished.")
        except subprocess.CalledProcessError as e:
            if e.returncode == 2:
                print(
                    (
                        f"\nOoops! PyTorch Geometric has not published a wheel for your PyTorch version yet: torch=={torch_ver} ({backend_tag}).\n"
                        f"Please follow our PyG troubleshooting guide for next steps: {DOC_URL}"
                    ),
                    file=sys.stderr,
                )
            else:
                raise e
    else:
        print("Dry-run: re-run with --run or -y to execute the command.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())