#!/usr/bin/env python3
"""Print the structure of the Franka stack cube HDF5 dataset."""

import argparse
from pathlib import Path

import h5py

# Project root = parent of isaac_envs (repo root when isaac_envs is inside the repo)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PATH = _PROJECT_ROOT / "isaac_envs" / "data" / "generated_dataset_small.hdf5"


def _visit(name: str, obj, indent: int, template_only: bool, demo_len: dict | None):
    prefix = "  " * indent
    if isinstance(obj, h5py.Dataset):
        shape_str = f"shape={obj.shape}, dtype={obj.dtype}"
        print(f"{prefix}{name}: {shape_str}")
        if demo_len is not None and name.count("/") == 2 and "demo_" in name:
            parts = name.split("/")
            if len(parts) == 3 and parts[0] == "data" and obj.ndim >= 1:
                key = parts[1]
                if key not in demo_len:
                    demo_len[key] = obj.shape[0]
    else:
        print(f"{prefix}{name}/ (Group)")
    if not isinstance(obj, h5py.Group):
        return
    # For template_only: recurse into data, then only into demo_0 (skip other demos)
    for k in sorted(obj.keys()):
        child_name = f"{name}/{k}" if name else k
        if template_only and name == "data" and k.startswith("demo_") and k != "demo_0":
            print(f"  {'  ' * (indent + 1)}{child_name}/ (Group) ...")
            continue
        _visit(child_name, obj[k], indent + 1, template_only, demo_len)


def main():
    parser = argparse.ArgumentParser(description="Print HDF5 dataset structure")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_PATH),
        help=f"HDF5 file path (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print every demo; default is template (data + demo_0) plus demo list",
    )
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f"Error: file not found: {path}")
        return 1

    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        print(f"Root keys: {list(f.keys())}")
        print()

        # Count demos and trajectory lengths
        if "data" in f:
            demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
            num_demos = len(demo_keys)
            demo_lens = {}
            for k in demo_keys:
                if "actions" in f["data"][k]:
                    demo_lens[k] = f["data"][k]["actions"].shape[0]
            print(f"data/: {num_demos} demos")
            if demo_lens:
                lengths = list(demo_lens.values())
                print(f"  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, total_steps={sum(lengths)}")
            print()

        print("Structure (template: data/demo_0 and its subgroups):")
        print("-" * 60)
        demo_len = {} if args.full else None
        _visit("data", f["data"], 0, template_only=not args.full, demo_len=demo_len)

        if not args.full and "data" in f:
            demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
            if len(demo_keys) > 1:
                print()
                print("All demo keys and lengths:")
                for k in demo_keys:
                    n = f["data"][k]["actions"].shape[0]
                    print(f"  {k}: {n} steps")

    return 0


if __name__ == "__main__":
    exit(main())
