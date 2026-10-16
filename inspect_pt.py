"""Show the attributes stored in the first graph of a PyTorch dataset."""

import argparse
from pathlib import Path

import torch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to a trusted .pt graph list")
    options = parser.parse_args(argv)
    graphs = torch.load(options.dataset, weights_only=False)
    print(f"Loaded {len(graphs)} graphs from {options.dataset}")
    if not graphs:
        return
    for name, value in graphs[0]:
        shape = getattr(value, "shape", "N/A")
        print(f"{name}: type={type(value)}, shape={shape}")


if __name__ == "__main__":
    main()
