"""Convert consecutive hot-aisle CSV snapshots into PyG graph sequences."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.preprocessing import LabelBinarizer
from torch_geometric.data import Data


STATE_COLUMNS = [
    "Velocity(m/s)_0", "Velocity(m/s)_1", "Velocity(m/s)_2",
    "Pressure(Pa)", "Temperature(C)",
]
AXES = (
    (0, 1), (0, -1),
    (1, 1), (1, -1),
    (2, 1), (2, -1),
)


def global_features(folder_name):
    match = re.match(r"c(\d+)s(\d+)", folder_name)
    if match is None:
        raise ValueError(
            f"Directory {folder_name!r} must begin with c<number>s<number>"
        )
    return torch.tensor(
        [[int(match.group(1)), int(match.group(2))]], dtype=torch.float32
    )


def directional_edges(points, distance_threshold=0.3):
    """Connect the nearest point in each axis direction, on the same grid line."""
    tree = cKDTree(points)
    connections = []
    for source, point in enumerate(points):
        for axis, sign in AXES:
            offset = np.zeros(3)
            offset[axis] = sign * distance_threshold
            _, destination = tree.query(
                point + offset, k=1, distance_upper_bound=distance_threshold
            )
            if destination >= len(points) or destination == source:
                continue
            other_axes = [index for index in range(3) if index != axis]
            if np.array_equal(points[destination, other_axes], point[other_axes]):
                connections.append((source, destination))
    if not connections:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(connections, dtype=torch.long).T.contiguous()


def graph_pair(current_csv, next_csv, globals_, distance_threshold):
    current = pd.read_csv(current_csv)
    following = pd.read_csv(next_csv)
    coordinates = current[["x", "y", "z"]].to_numpy(dtype=np.float32)
    node_types = LabelBinarizer().fit_transform(current["boundary"].to_numpy())
    return Data(
        x=torch.tensor(current[STATE_COLUMNS].to_numpy(), dtype=torch.float32),
        y=torch.tensor(following[STATE_COLUMNS].to_numpy(), dtype=torch.float32),
        q_0=torch.tensor(coordinates, dtype=torch.float32),
        n=torch.tensor(node_types, dtype=torch.float32),
        edge_index=directional_edges(coordinates, distance_threshold),
        g=globals_,
    )


def convert_folder(directory, output_directory, distance_threshold=0.3):
    files = sorted(directory.glob("*.csv"))
    globals_ = global_features(directory.name)
    graphs = [
        graph_pair(current, following, globals_, distance_threshold)
        for current, following in zip(files[:-1], files[1:])
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / f"cylinder3d_{directory.name}.pt"
    torch.save(graphs, destination)
    return destination, len(graphs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder containing c<number>s<number> folders")
    parser.add_argument("--output-dir", type=Path, help="Defaults to input_dir")
    parser.add_argument("--distance-threshold", type=float, default=0.3)
    args = parser.parse_args(argv)
    destination = args.output_dir or args.input_dir
    for folder in args.input_dir.iterdir():
        if folder.is_dir():
            path, count = convert_folder(folder, destination, args.distance_threshold)
            print(f"Saved {count} graph pairs to {path}")


if __name__ == "__main__":
    main()
