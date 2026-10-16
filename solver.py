"""Training, evaluation, and rollout of the thermodynamic graph model."""

from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import load_dataset
from model import PIGNN3D
from utilities.plot import plot_2D, plot_3D, save_simulation_to_vtk
from utilities.utils import print_error, save_log


LOSS_NAMES = ("loss_z", "loss_deg_E", "loss_deg_S")
ERROR_FIELDS = {
    "couette": {"q": (0, 1), "v": (2,), "e": (3,), "tau": (4,)},
    "beam": {"q": (0, 1, 2), "v": (3, 4, 5), "sigma": tuple(range(6, 12))},
    "cylinder": {"v": (0, 1), "P": (2,)},
    "cylinder3d": {"v": (0, 1, 2), "P": (3,), "T": (4,)},
}


class Solver:
    def __init__(self, args):
        self.args = args
        self.sys_name = args.sys_name
        self.device = torch.device(
            "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
        )
        self.train_set, self.val_set, self.test_set = load_dataset(args)
        self.dims = self.train_set.dims
        self.dt = self.train_set.dt
        self.stats_z, self.stats_q = self.train_set.get_stats(self.device)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.net = PIGNN3D(args, self.dims).to(self.device)
        if not args.train:
            checkpoint = Path(args.dset_dir) / f"params_{self.sys_name}.pt"
            self.net.load_state_dict(torch.load(checkpoint, map_location=self.device))

        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=args.miles, gamma=args.gamma
        )

    @staticmethod
    def norm(values, stats):
        return (values - stats["mean"]) / stats["std"]

    @staticmethod
    def denorm(values, stats):
        return values * stats["std"] + stats["mean"]

    @staticmethod
    def integrator(l_matrix, m_matrix, energy_gradient, entropy_gradient):
        # GENERIC evolution with the two degeneracy residuals.
        derivative = torch.bmm(l_matrix, energy_gradient) + torch.bmm(
            m_matrix, entropy_gradient
        )
        energy_residual = torch.bmm(m_matrix, energy_gradient)
        entropy_residual = torch.bmm(l_matrix, entropy_gradient)
        return derivative[:, :, 0], energy_residual[:, :, 0], entropy_residual[:, :, 0]

    def _batch_features(self, snapshots, state=None):
        snapshots = snapshots.to(self.device)
        z = self.norm(snapshots.x, self.stats_z) if state is None else state
        target = self.norm(snapshots.y, self.stats_z)
        keys = snapshots.keys()
        auxiliary = (
            self.norm(snapshots.q_0, self.stats_q) if "q_0" in keys else None
        )
        forces = self.norm(snapshots.f, self.stats_q) if "f" in keys else None
        globals_ = snapshots.g if "g" in keys else None
        batch = snapshots.batch if "batch" in keys else None
        return snapshots, z, target, auxiliary, forces, globals_, batch

    def _losses(self, snapshots, z, target, auxiliary, forces, globals_, batch):
        operators = self.net(
            z, snapshots.n, snapshots.edge_index,
            q_0=auxiliary, f=forces, g=globals_, batch=batch,
        )
        predicted, energy_residual, entropy_residual = self.integrator(*operators)
        derivative = (target - z) / self.dt
        interior = snapshots.n[:, 0] == 1
        return (
            ((derivative - predicted) ** 2)[interior].mean(),
            (energy_residual ** 2)[interior].mean(),
            (entropy_residual ** 2)[interior].mean(),
        )

    def _run_epoch(self, dataset, training):
        totals = torch.zeros(3)
        batches = 0
        simulations = tqdm(range(len(dataset)), ncols=100) if training else range(len(dataset))
        for simulation in simulations:
            loader = DataLoader(
                dataset[simulation],
                batch_size=self.args.batch_size,
                shuffle=training,
                num_workers=4,
            )
            for snapshots in loader:
                items = self._batch_features(snapshots)
                snapshots, z, target, auxiliary, forces, globals_, batch = items
                if training:
                    # Noise is applied only to the interior state before differentiation.
                    interior = snapshots.n[:, 0] == 1
                    z = z.clone()
                    z[interior] += self.args.noise_var ** 0.5 * torch.randn_like(z[interior])
                losses = self._losses(
                    snapshots, z, target, auxiliary, forces, globals_, batch
                )
                if training:
                    objective = self.args.lambda_d * losses[0] + losses[1] + losses[2]
                    self.optim.zero_grad()
                    objective.backward()
                    self.optim.step()
                totals += torch.tensor([item.item() for item in losses])
                batches += 1
        if batches == 0:
            raise ValueError("A dataset split contains no graph batches.")
        return (totals / batches).tolist()

    def train_model(self):
        logs = {
            phase: {"epoch": [], **{name: [] for name in LOSS_NAMES}}
            for phase in ("train", "val")
        }
        print("\n[Training Started]\n")
        for epoch in range(1, self.args.max_epoch + 1):
            print(f"[Epoch: {epoch}]")
            self.net.train()
            training = self._run_epoch(self.train_set, training=True)
            self.scheduler.step()
            self.net.eval()
            validation = self._run_epoch(self.val_set, training=False)
            for phase, values in (("train", training), ("val", validation)):
                logs[phase]["epoch"].append(epoch)
                for name, value in zip(LOSS_NAMES, values):
                    logs[phase][name].append(value)
            for label, train_value, val_value in zip(
                ("Data Loss", "Deg Loss (E)", "Deg Loss (S)"), training, validation
            ):
                print(f"{label}: {train_value:1.2e} (Train) / {val_value:1.2e} (Val)")
            print()

        print("[Training Finished]\n")
        torch.save(self.net.state_dict(), self.output_dir / f"params_{self.sys_name}.pt")
        for phase, values in logs.items():
            save_log(self.args, values, phase)

    def test_model(self):
        self.net.eval()
        print("[Test Set Evaluation]")
        print_error(self.compute_error(self.test_set))
        print("[Test Evaluation Finished]\n")

    def plot_sim(self, sim=8):
        data_list = self.train_set[sim]
        predictions, truth = self.integrate_sim(data_list)
        if self.sys_name == "beam":
            plot_3D(predictions, truth, data_list, self.output_dir)
        elif self.sys_name == "cylinder":
            plot_2D(predictions, truth, data_list, self.output_dir)
        elif self.sys_name == "cylinder3d":
            save_simulation_to_vtk(
                predictions, data_list, self.output_dir, input_number=self.args.cooling
            )

    def compute_error(self, dataset):
        errors = {name: [] for name in ERROR_FIELDS[self.sys_name]}
        for simulation in dataset:
            prediction, truth = self.integrate_sim(simulation)
            difference = (prediction[1:] - truth[1:]).numpy()
            reference = truth[1:].numpy()
            for name, columns in ERROR_FIELDS[self.sys_name].items():
                numerator = (difference[:, :, list(columns)] ** 2).sum((1, 2))
                denominator = (reference[:, :, list(columns)] ** 2).sum((1, 2))
                errors[name].extend((numerator / denominator) ** 0.5)
        return errors

    def integrate_sim(self, data_list, full_rollout=True):
        nodes, state_width = data_list[0].x.size(0), self.dims["z"]
        prediction = torch.zeros(len(data_list) + 1, nodes, state_width)
        truth = torch.zeros_like(prediction)
        prediction[0] = data_list[0].x
        truth[0] = data_list[0].x
        state = self.norm(data_list[0].x.to(self.device), self.stats_z)

        for index, snapshot in enumerate(DataLoader(data_list)):
            snapshot, _, target, auxiliary, forces, globals_, batch = (
                self._batch_features(snapshot, state)
            )
            operators = self.net(
                state, snapshot.n, snapshot.edge_index,
                q_0=auxiliary, f=forces, g=globals_, batch=batch,
            )
            derivative, _, _ = self.integrator(*operators)
            next_state = state + self.dt * derivative
            # Boundary nodes receive their measured next state at each time step.
            for boundary in range(1, snapshot.n.size(1)):
                mask = snapshot.n[:, boundary] == 1
                next_state[mask] = target[mask]
            prediction[index + 1] = self.denorm(
                next_state.detach(), self.stats_z
            ).cpu()
            truth[index + 1] = snapshot.y.cpu()
            state = next_state.detach() if full_rollout else target
        return prediction, truth
