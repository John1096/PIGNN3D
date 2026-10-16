# TiGNN 3D

Research code for a thermodynamics-informed graph neural network. The model
encodes node and edge features, performs graph message passing, and predicts
the energy and entropy gradients plus GENERIC operators used to advance the
state.

This repository is a cleaned-up version of the `tignn3d_tmp_grad` source.
The training equations, dataset splits, and visualization outputs remain
recognizable. The CSV converter now accepts an input path and writes its
two global case features correctly.

## Contents

- `main.py`: command-line entry point
- `model.py`: graph network and thermodynamic decoders
- `dataset.py`: simulation loaders and normalization
- `solver.py`: training, evaluation, and rollout
- `graphcon.py`: CSV snapshots to PyG graph sequences
- `data/update.py`: boundary correction for source CSVs
- `utilities/plot.py`: VTK, Excel, and animation outputs
- `inspect_pt.py`: inspect a serialized graph sequence

## Setup

Create a Python environment with a PyTorch build suitable for your system,
then install the dependencies in `requirements.txt`. PyTorch Geometric and
`torch-scatter` may require builds that match the installed PyTorch and CUDA
versions.

## Data

The original dataset and trained checkpoints are not included. Put trusted
serialized graph sequences under `data/database_cylinder3d/`. The loader
accepts `cylinder3d1.pt` through `cylinder3d13.pt`, or the sorted
`cylinder3d_c*s*.pt` files produced by `graphcon.py`. The latter naming
scheme uses alphabetical order to map files to simulation indices; check that
this order matches your intended cases before training.

To convert CSV snapshots:

```shell
python graphcon.py path/to/hot_aisle --output-dir data/database_cylinder3d
```

Each case folder must start with a name like `c40s100` and contain sequential
CSV snapshots. The converter saves successive frame pairs as PyG graphs.
Only load `.pt` files from sources you trust, because they are deserialized
with `torch.load`.

## Run

```shell
python main.py --dset_dir data --output_dir outputs
```

The default study is `cylinder3d`, with training, testing, and VTK export
enabled. Run `python main.py --help` for all settings. For evaluation only,
place `params_cylinder3d.pt` in the dataset directory and pass
`--train false`.

The code was checked for Python syntax, but an end-to-end training run requires
the original data and scientific Python dependencies.
