"""Dataset layouts and simulation-level train/validation/test splits."""

import itertools
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


def _batch_statistics(dataset, field, device):
    """Match the source normalization: average moments over simulations/nodes."""
    total = 0
    for simulation in dataset:
        batch = Batch.from_data_list(simulation)
        total += getattr(batch, field).sum(0)
    count = len(dataset) * len(batch.batch)
    mean = total / count

    squared_error = 0
    for simulation in dataset:
        batch = Batch.from_data_list(simulation)
        squared_error += ((getattr(batch, field) - mean) ** 2).sum(0)
    std = (squared_error / count).sqrt()
    std[std == 0] = 1
    return {"mean": mean.to(device), "std": std.to(device)}


class SimulationDataset(Dataset):
    dims = None
    dt = None

    def __init__(self, simulations, directory):
        self.sims = simulations
        self.dset_dir = Path(directory)

    def __len__(self):
        return len(self.sims)

    def filename(self, simulation):
        raise NotImplementedError

    def __getitem__(self, index):
        # Each serialized file stores all graph snapshots for one simulation.
        return torch.load(
            self.dset_dir / self.filename(self.sims[index]), weights_only=False
        )

    def get_stats(self, device):
        state = _batch_statistics(self, "y", device)
        auxiliary = self.auxiliary_field()
        return state, (_batch_statistics(self, auxiliary, device) if auxiliary else None)

    def auxiliary_field(self):
        return None


class CouetteDataset(SimulationDataset):
    dims = {"z": 5, "q": 2, "q_0": 0, "n": 2, "f": 0, "g": 2}
    dt = 1 / 150

    def filename(self, simulation):
        reynolds, weissenberg = simulation
        return f"couette_Re_{reynolds:.1f}_We_{weissenberg:.1f}.pt"


class BeamDataset(SimulationDataset):
    dims = {"z": 12, "q": 3, "q_0": 0, "n": 2, "f": 3, "g": 0}
    dt = 1 / 20

    def filename(self, simulation):
        return f"beam_{simulation[0] + 1}.pt"

    def auxiliary_field(self):
        return "f"


class CylinderDataset(SimulationDataset):
    dims = {"z": 3, "q": 0, "q_0": 2, "n": 4, "f": 0, "g": 0}
    dt = 1 / 100

    def filename(self, simulation):
        position, velocity = simulation
        return f"cylinder_{position}_v_{velocity:.2f}.pt"

    def auxiliary_field(self):
        return "q_0"


class CylinderDataset3D(CylinderDataset):
    # graphcon.py stores cooling/server settings as two global features.
    dims = {"z": 5, "q": 0, "q_0": 3, "n": 4, "f": 0, "g": 2}
    dt = 1 / 200

    def __init__(self, simulations, directory):
        super().__init__(simulations, directory)
        self.dims = self.dims.copy()
        if simulations:
            # Older graph files omitted g; match the actual graph schema.
            first_graph = self[0][0]
            self.dims["g"] = first_graph.g.numel() if "g" in first_graph.keys() else 0

    def filename(self, simulation):
        numbered = f"cylinder3d{simulation[0] + 1}.pt"
        if (self.dset_dir / numbered).exists():
            return numbered
        # Fresh graphcon.py output is named after each c...s... case.
        generated = sorted(self.dset_dir.glob("cylinder3d_c*s*.pt"))
        if simulation[0] < len(generated):
            return generated[simulation[0]].name
        return numbered


SYSTEMS = {
    "couette": (
        CouetteDataset,
        ([0.1 * index for index in range(1, 11)],
         [1 + 0.1 * index for index in range(11)]),
    ),
    "beam": (BeamDataset, (list(range(52)),)),
    "cylinder": (
        CylinderDataset,
        ([1, 2, 3, 4, 5, 6], [1, 1.25, 1.5, 1.75, 2]),
    ),
    "cylinder3d": (CylinderDataset3D, (list(range(13)),)),
}


def split_dataset(*parameter_values):
    simulations = list(itertools.product(*parameter_values))
    np.random.shuffle(simulations)
    count = len(simulations)
    training_end = int(0.8 * count)
    validation_end = int(0.9 * count)
    return (
        simulations[:training_end],
        simulations[training_end:validation_end],
        simulations[validation_end:],
    )


def load_dataset(args):
    try:
        dataset_class, parameters = SYSTEMS[args.sys_name]
    except KeyError as error:
        raise ValueError(f"Unknown system: {args.sys_name}") from error
    directory = Path(args.dset_dir) / f"database_{args.sys_name}"
    splits = split_dataset(*parameters)
    return tuple(dataset_class(simulations, directory) for simulations in splits)
