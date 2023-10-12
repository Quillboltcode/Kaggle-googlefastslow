import torch
from torch.nn import Linear, Dropout, BatchNorm1d, LeakyReLU
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    SAGEConv,
    GATv2Conv,
    MemPooling,
    GATConv,
    DeepGCNLayer,
)
import torch.nn.functional as F
from torch import nn
from torch import Tensor , fmod


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            xs = []
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
            xs.append(x)
            if i == 0:
                x_all = torch.cat(xs, dim=0)
                layer_1_embeddings = x_all
            elif i == 1:
                x_all = torch.cat(xs, dim=0)
                layer_2_embeddings = x_all
            elif i == 2:
                x_all = torch.cat(xs, dim=0)
                layer_3_embeddings = x_all
        # return x.log_softmax(dim=-1)
        return layer_1_embeddings, layer_2_embeddings, layer_3_embeddings


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_channels, graph_in, graph_out, hidden_dim, dropout=0.0):
        super().__init__()
        op_embedding_dim = 4  # I choose 4-dimensional embedding
        self.embedding = torch.nn.Embedding(
            120,  # 120 different op-codes
            op_embedding_dim,
        )
        assert len(hidden_channels) > 0

        self.linear = nn.Linear(op_embedding_dim + 140, graph_in)
        in_channels = graph_in
        self.convs = torch.nn.ModuleList()
        last_dim = hidden_channels[0]
        conv = SAGEConv
        self.convs.append(conv(in_channels, hidden_channels[0]))
        for i in range(len(hidden_channels) - 1):
            self.convs.append(conv(hidden_channels[i], hidden_channels[i + 1]))
            last_dim = hidden_channels[i + 1]
        self.convs.append(conv(last_dim, graph_out))

        self.dense = torch.nn.Sequential(
            nn.Linear(graph_out * 2 + 24, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    #         self.dropout = nn.Dropout(p=dropout)

    def forward(
        self, x_cfg: Tensor, x_feat: Tensor, x_op: Tensor, edge_index: Tensor
    ) -> Tensor:
        # get graph features
        x = torch.concat([x_feat, self.embedding(x_op)], dim=1)
        x = self.linear(x)
        # pass though conv layers
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        # get 1d graph embedding using average pooling
        x_mean = x.mean(0)
        x_max = x.max(0).values

        # put graph data into config data
        x = torch.concat(
            [x_cfg, x_max.repeat((len(x_cfg), 1)), x_mean.repeat((len(x_cfg), 1))],
            axis=1,
        )
        # put into dense nn
        x = torch.flatten(self.dense(x))
        x = (x - torch.mean(x)) / (torch.std(x) + 1e-5)
        return x


class GATWithMemPooling(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, hidden_dim, dropout):
        super().__init__()
        op_embedding_dim = 4  # I choose 4-dimensional embedding
        self.embedding = torch.nn.Embedding(
            120,  # 120 different op-codes
            op_embedding_dim,
        )
        assert len(hidden_channels) > 0
        self.dropout = dropout

        self.linear = Linear(op_embedding_dim + 140, in_channels)

        self.convs = torch.nn.ModuleList()
        for i in range(len(hidden_channels) - 1):
            conv = GATv2Conv(hidden_channels[i], hidden_channels[i + 1])
            norm = BatchNorm1d(hidden_channels[i + 1])
            act = LeakyReLU()
            self.convs.append(
                DeepGCNLayer(conv, norm, act, block="res+", dropout=dropout)
            )

        self.mem1 = MemPooling(hidden_channels[-1], 80, heads=5, num_clusters=10)
        self.mem2 = MemPooling(80, out_channels, heads=5, num_clusters=1)

        self.dense = torch.nn.Sequential(
            nn.Linear(out_channels * 2 + 24, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, batch):
        x = self.lin(x)
        for conv in self.convs:
            x = conv(x, edge_index)

        x, S1 = self.mem1(x, batch)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout)
        x, S2 = self.mem2(x)

        return (
            F.log_softmax(x.squeeze(1), dim=-1),
            MemPooling.kl_loss(S1) + MemPooling.kl_loss(S2),
        )


if __name__ == "__main__":
    # model1 = SAGE(in_channels=64, hidden_channels=64, out_channels=64, num_layers=3)
    model1 = SimpleModel(
        hidden_channels=[32, 48, 64, 84],
        graph_in=64,
        graph_out=64,
        hidden_dim=128,
        dropout=0.2,
    )
    # model2 = SAGE(in_channels=64, hidden_channels=64, out_channels=64)
    model2 = GATWithMemPooling(
        in_channels=64,
        hidden_channels=[32, 48, 64, 84],
        out_channels=64,
        hidden_dim=128,
        dropout=0.2,
    )
    # x = torch.randn(64, 64)
    # adjs = [torch.randn(2, 3), torch.randn(2, 3)]
    # model1(x, adjs)
    # model2(x, adjs)
    print(model1)
    print(model2)
