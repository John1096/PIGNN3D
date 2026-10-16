"""Graph network that predicts the GENERIC operators and state gradients."""

import torch
from torch import nn
from torch_scatter import scatter_add


class MLP(nn.Module):
    """Linear layers with SiLU activations between hidden layers."""

    def __init__(self, widths):
        super().__init__()
        layers = []
        for index, (input_size, output_size) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
        self.layers = nn.ModuleList(layers)

    def forward(self, values):
        for layer in self.layers:
            values = layer(values)
        return values


class EdgeModel(nn.Module):
    def __init__(self, args, dims):
        super().__init__()
        width = args.dim_hidden
        self.edge_mlp = MLP([3 * width + dims["g"]] + [width] * args.n_hidden + [width])

    def forward(self, source, destination, edge_features, globals_=None, batch=None):
        features = [edge_features, source, destination]
        if globals_ is not None:
            features.append(globals_[batch])
        return self.edge_mlp(torch.cat(features, dim=1))


class NodeModel(nn.Module):
    def __init__(self, args, dims):
        super().__init__()
        width = args.dim_hidden
        self.node_mlp = MLP(
            [2 * width + dims["f"] + dims["g"]]
            + [width] * args.n_hidden
            + [width]
        )

    def forward(self, nodes, node_types, edge_index, edge_features,
                forces=None, globals_=None, batch=None):
        # Messages accumulate at destination nodes; boundary nodes keep their state.
        destinations = edge_index[1]
        messages = scatter_add(edge_features, destinations, dim=0, dim_size=nodes.size(0))
        features = [nodes, messages]
        if forces is not None:
            features.append(forces)
        elif globals_ is not None:
            features.append(globals_[batch])
        proposal = self.node_mlp(torch.cat(features, dim=1))
        updated = nodes.clone()
        interior = node_types[:, 0] == 1
        updated[interior] = proposal[interior]
        return updated


class MetaLayer(nn.Module):
    """One edge update followed by one node update."""

    def __init__(self, edge_model=None, node_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model

    def forward(self, nodes, node_types, edge_index, edge_features,
                forces=None, globals_=None, batch=None):
        sources = edge_index[0]
        destinations = edge_index[1]
        edge_batch = batch if batch is None else batch[sources]
        edge_features = self.edge_model(
            nodes[sources], nodes[destinations], edge_features, globals_, edge_batch
        )
        nodes = self.node_model(
            nodes, node_types, edge_index, edge_features, forces, globals_, batch
        )
        return nodes, edge_features


class PIGNN3D(nn.Module):
    """Encode a graph, pass messages, and decode thermodynamic quantities."""

    def __init__(self, args, dims):
        super().__init__()
        self.dims = dims
        self.mixp = args.mixp
        self.dim_z = dims["z"]
        self.dim_q = dims["q"]
        width = args.dim_hidden
        hidden = [width] * args.n_hidden

        node_inputs = dims["z"] + dims["n"] - dims["q"]
        edge_inputs = dims["q"] + dims["q_0"] + 1
        self.encoder_node = MLP([node_inputs] + hidden + [width])
        self.encoder_edge = MLP([edge_inputs] + hidden + [width])
        self.processor = nn.ModuleList(
            [MetaLayer(EdgeModel(args, dims), NodeModel(args, dims))
             for _ in range(args.passes)]
        )

        self.decoder_E = MLP([width] + hidden + [self.dim_z])
        self.decoder_S = MLP([width] + hidden + [self.dim_z])
        self.decoder_L = MLP([width] + hidden + [self.dim_z * (self.dim_z - 1) // 2])
        self.decoder_M = MLP([width] + hidden + [self.dim_z * (self.dim_z + 1) // 2])

    def forward(self, z, n, edge_index, q_0=None, f=None, g=None, batch=None):
        z.requires_grad_(True)
        positions = q_0 if q_0 is not None else z[:, :self.dim_q]
        state = z if q_0 is not None else z[:, self.dim_q:]
        nodes = self.encoder_node(torch.cat((state, n), dim=1))

        source, destination = edge_index
        displacement = positions[source] - positions[destination]
        distance = torch.norm(displacement, dim=1, keepdim=True)
        edges = self.encoder_edge(torch.cat((displacement, distance), dim=1))

        for layer in self.processor:
            node_delta, edge_delta = layer(nodes, n, edge_index, edges, f, g, batch)
            nodes = nodes + node_delta
            edges = edges + edge_delta

        energy_gradient = self.decoder_E(nodes).unsqueeze(2)
        entropy_gradient = self.decoder_S(nodes).unsqueeze(2)
        lower_l = self.decoder_L(nodes)
        lower_m = self.decoder_M(nodes)

        # Antisymmetry of L and the Gram product for M enforce GENERIC structure.
        shape = (nodes.size(0), self.dim_z, self.dim_z)
        l_matrix = lower_l.new_zeros(shape)
        m_factor = lower_m.new_zeros(shape)
        lower_off_diagonal = torch.tril_indices(self.dim_z, self.dim_z, offset=-1)
        lower_diagonal = torch.tril_indices(self.dim_z, self.dim_z)
        l_matrix[:, lower_off_diagonal[0], lower_off_diagonal[1]] = lower_l
        m_factor[:, lower_diagonal[0], lower_diagonal[1]] = lower_m
        l_matrix = l_matrix - l_matrix.transpose(1, 2)
        m_matrix = torch.bmm(m_factor, m_factor.transpose(1, 2))
        return l_matrix, m_matrix, energy_gradient, entropy_gradient
