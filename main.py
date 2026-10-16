"""Train, evaluate, and export thermodynamics-informed graph predictions."""

import argparse

import numpy as np
import torch

from solver import Solver
from utilities.utils import str2bool


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    study = parser.add_argument_group("study")
    study.add_argument("--sys_name", default="cylinder3d")
    study.add_argument("--train", type=str2bool, default=True)
    study.add_argument("--gpu", type=str2bool, default=True)

    data = parser.add_argument_group("data")
    data.add_argument("--dset_dir", default="data")
    data.add_argument("--test_dir", default="data")
    data.add_argument("--cooling", default="95")

    network = parser.add_argument_group("network")
    network.add_argument("--n_hidden", type=int, default=2)
    network.add_argument("--dim_hidden", type=int, default=10)
    network.add_argument("--passes", type=int, default=10)
    network.add_argument("--skip", type=int, default=100)

    training = parser.add_argument_group("training")
    training.add_argument("--seed", type=int, default=1)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--lambda_d", type=float, default=1e2)
    training.add_argument("--noise_var", type=float, default=1e-2)
    training.add_argument("--batch_size", type=int, default=2)
    training.add_argument("--max_epoch", type=int, default=6000)
    training.add_argument("--miles", type=int, nargs="+", default=[2000, 4000])
    training.add_argument("--gamma", type=float, default=1e-1)
    training.add_argument("--mixp", type=str2bool, default=False)

    output = parser.add_argument_group("output")
    output.add_argument("--output_dir", default="outputs")
    output.add_argument("--plot_sim", type=str2bool, default=True)
    return parser


def main(args=None):
    options = build_parser().parse_args(args)
    torch.manual_seed(options.seed)
    np.random.seed(options.seed)

    solver = Solver(options)
    if options.train:
        solver.train_model()
    solver.test_model()
    if options.plot_sim:
        solver.plot_sim()


if __name__ == "__main__":
    main()
