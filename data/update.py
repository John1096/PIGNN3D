"""Apply the outlet boundary correction to simulation CSV files."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "y": ("y", "Y", "Points_1"),
    "z": ("z", "Z", "Points_2"),
    "boundary": ("boundary", "Boundary", "BOUNDARY"),
    "vx": ("Velocity(m/s)_0", "velocity(m/s)_0", "Velocity_0", "velocity_x", "vx"),
    "vy": ("Velocity(m/s)_1", "velocity(m/s)_1", "Velocity_1", "velocity_y", "vy"),
    "vz": ("Velocity(m/s)_2", "velocity(m/s)_2", "Velocity_2", "velocity_z", "vz"),
}


def resolve_columns(frame):
    columns = {
        field: next((name for name in aliases if name in frame.columns), None)
        for field, aliases in COLUMN_ALIASES.items()
    }
    missing = [field for field, column in columns.items() if column is None]
    return columns, missing


def process_csv(path, tolerance=1e-6):
    try:
        frame = pd.read_csv(path)
    except Exception as error:
        print(f"[SKIP] Failed to read {path}: {error}", file=sys.stderr)
        return 0, 0

    columns, missing = resolve_columns(frame)
    if missing:
        print(f"[WARN] {path.name}: missing columns {missing}; no changes made")
        return 0, 0
    for field in ("y", "z"):
        column = columns[field]
        if not np.issubdtype(frame[column].dtype, np.number):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Select the thin outlet strip; compare z with a tolerance for CSV rounding.
    y = frame[columns["y"]]
    z = frame[columns["z"]]
    mask = (y > 2.05) & (y < 2.73) & np.isfinite(z) & (np.abs(z - 4.5) <= tolerance)
    affected = int(mask.sum())
    if affected:
        frame.loc[mask, columns["boundary"]] = 4
        frame.loc[mask, [columns["vx"], columns["vy"], columns["vz"]]] = 0
        try:
            frame.to_csv(path, index=False)
            print(f"[OK] {path.name}: updated {affected} rows")
        except Exception as error:
            print(f"[ERR] Failed to write {path}: {error}", file=sys.stderr)
    else:
        print(f"[SKIP] {path.name}: 0 rows matched condition")
    return len(frame), affected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--tolerance", type=float, default=1e-6)
    options = parser.parse_args(argv)
    files = sorted(
        csv
        for folder in options.root.iterdir() if folder.is_dir()
        for csv in folder.iterdir() if csv.is_file() and csv.suffix.lower() == ".csv"
    )
    total_rows = total_affected = 0
    for csv in files:
        rows, affected = process_csv(csv, options.tolerance)
        total_rows += rows
        total_affected += affected
    print(
        f"CSV files scanned: {len(files)}\n"
        f"Total rows scanned: {total_rows}\n"
        f"Total rows updated: {total_affected}"
    )


if __name__ == "__main__":
    main()
