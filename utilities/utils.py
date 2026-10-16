"""Command-line parsing and reporting helpers."""

import argparse
from pathlib import Path


TRUE_VALUES = {"yes", "true", "t", "y", "1"}
FALSE_VALUES = {"no", "false", "f", "n", "0"}


def str2bool(value):
    normalized = value.lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def save_log(args, log, phase):
    destination = Path(args.output_dir) / f"{phase}_log_{args.sys_name}.txt"
    columns = ("epoch", "loss_z", "loss_deg_E", "loss_deg_S")
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(columns) + "\n")
        for row in zip(*(log[column] for column in columns)):
            handle.write(" ".join(map(str, row)) + "\n")


def print_error(errors):
    print("State Variable (L2 relative error)")
    for variable, values in errors.items():
        print(f"  {variable} = {sum(values) / len(values):1.2e}")
