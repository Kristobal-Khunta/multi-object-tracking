import torch
from torch import nn
from torch.nn import functional as F
from ..utils import ltrb_to_xcycwh, cosine_distance


class BipartiteNeuralMessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, dropout=0.0):
        super().__init__()

        edge_in_dim = (
            2 * node_dim + 2 * edge_dim
        )  # 2*edge_dim since we always concatenate initial edge features
        self.edge_mlp = nn.Sequential(
            *[
                nn.Linear(edge_in_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

        node_in_dim = node_dim + edge_dim
        self.node_mlp = nn.Sequential(
            *[
                nn.Linear(node_in_dim, node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(node_dim, node_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )

    def edge_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Node-to-edge updates, as descibed in slide 71, lecture 5.
        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, 2 x edge_dim)
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_a_embeds: torch.Tensor with shape (|B|, node_dim)

        returns:
            updated_edge_feats = torch.Tensor with shape (|A|, |B|, edge_dim)
        """
        n_nodes_a, n_nodes_b, _ = edge_embeds.shape
        nodes_a_in = nodes_a_embeds.unsqueeze(1).expand((n_nodes_a, n_nodes_b, -1))
        nodes_b_in = nodes_b_embeds.unsqueeze(0).expand((n_nodes_a, n_nodes_b, -1))

        # edge_in has shape (|A|, |B|, 2*node_dim + 2*edge_dim)
        edge_in = torch.cat((nodes_a_in, edge_embeds, nodes_b_in), dim=-1)
        return self.edge_mlp(edge_in)

    def node_update(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        """
        Edge-to-node updates, as descibed in slide 75, lecture 5.

        Args:
            edge_embeds: torch.Tensor with shape (|A|, |B|, edge_dim)
            nodes_a_embeds: torch.Tensor with shape (|A|, node_dim)
            nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)

        returns:
            tuple(
                updated_nodes_a_embeds: torch.Tensor with shape (|A|, node_dim),
                updated_nodes_b_embeds: torch.Tensor with shape (|B|, node_dim)
                )
        """

        # Use 'sum' as aggregation function
        # aggreagete information about all connections of node A
        # in each row - sum over edge embeddings with neighborn
        nodes_a_neigh_embeds = torch.sum(
            edge_embeds, axis=1
        )  # shape (|A|, |B|, edge_dim) sum over B
        nodes_b_neigh_embeds = torch.sum(
            edge_embeds, axis=0
        )  # shape (|A|, |B|, edge_dim) sum over A
        nodes_a_in = torch.cat(
            (nodes_a_embeds, nodes_a_neigh_embeds), dim=-1
        )  # Has shape (|A|, node_dim + edge_dim)
        nodes_b_in = torch.cat(
            (
                nodes_b_embeds,
                nodes_b_neigh_embeds,
            ),
            dim=-1,
        )  # Has shape (|B|, node_dim + edge_dim)

        nodes_a = self.node_mlp(nodes_a_in)
        nodes_b = self.node_mlp(nodes_b_in)

        return nodes_a, nodes_b

    def forward(self, edge_embeds, nodes_a_embeds, nodes_b_embeds):
        edge_embeds_latent = self.edge_update(
            edge_embeds, nodes_a_embeds, nodes_b_embeds
        )
        nodes_a_latent, nodes_b_latent = self.node_update(
            edge_embeds_latent, nodes_a_embeds, nodes_b_embeds
        )

        return edge_embeds_latent, nodes_a_latent, nodes_b_latent


class SimilarityNet(nn.Module):
    def __init__(
        self,
        reid_network,
        node_dim,
        edge_dim,
        reid_dim,
        edges_in_dim,
        num_steps,
        dropout=0.0,
    ):
        super().__init__()
        self.reid_network = reid_network
        self.graph_net = BipartiteNeuralMessagePassingLayer(
            node_dim=node_dim, edge_dim=edge_dim, dropout=dropout
        )
        self.num_steps = num_steps
        self.cnn_linear = nn.Linear(reid_dim, node_dim)
        self.edge_in_mlp = nn.Sequential(
            *[
                nn.Linear(edges_in_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_dim, edge_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        self.classifier = nn.Sequential(
            *[nn.Linear(edge_dim, edge_dim), nn.ReLU(), nn.Linear(edge_dim, 1)]
        )

    @staticmethod
    def compute_edge_feats(track_coords, current_coords, track_t, curr_t):
        """
        Computes initial edge feature tensor

        Args:
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)

            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)


        Returns:
            tensor with shape (num_trakcs, num_boxes, 5) containing pairwise
            position and time difference features
        """
        num_boxes = current_coords.shape[0]
        num_tracks = track_coords.shape[0]

        track_coords = ltrb_to_xcycwh(track_coords)
        current_coords = ltrb_to_xcycwh(current_coords)

        track_coords = track_coords.unsqueeze_(1).expand(num_tracks, num_boxes, 4)
        current_coords = current_coords.unsqueeze_(0).expand(num_tracks, num_boxes, 4)

        dist_y = track_coords[..., 1] - current_coords[..., 1]
        dist_x = track_coords[..., 0] - current_coords[..., 0]
        denom = (track_coords[..., 2] + current_coords[..., 2]) / 2
        dist_x = dist_x / denom
        dist_y = dist_y / denom

        dist_w = torch.log(current_coords[..., 2] / track_coords[..., 2])
        dist_h = torch.log(current_coords[..., 3] / track_coords[..., 3])

        curr_t = curr_t.unsqueeze(0)  # .expand(num_tracks,num_boxes)
        track_t = track_t.unsqueeze(1)  # .expand(num_tracks, num_boxes)
        dist_t = (curr_t - track_t).type(dist_h.dtype)

        edge_feats = torch.stack([dist_x, dist_y, dist_w, dist_h, dist_t], dim=-1)

        return edge_feats  # has shape (num_trakcs, num_boxes, 5)

    def forward(
        self, track_app, current_app, track_coords, current_coords, track_t, curr_t
    ):
        """
        Args:
            track_app: track's reid embeddings, torch.Tensor with shape (num_tracks, 512)
            current_app: current frame detections' reid embeddings, torch.Tensor with shape (num_boxes, 512)
            track_coords: track's frame box coordinates, given by top-left and bottom-right coordinates
                          torch.Tensor with shape (num_tracks, 4)
            current_coords: current frame box coordinates, given by top-left and bottom-right coordinates
                            has shape (num_boxes, 4)

            track_t: track's timestamps, torch.Tensor with with shape (num_tracks, )
            curr_t: current frame's timestamps, torch.Tensor withwith shape (num_boxes,)

        Returns:
            classified edges: torch.Tensor with shape (num_steps, num_tracks, num_boxes),
                             containing at entry (step, i, j) the unnormalized probability that track i and
                             detection j are a match, according to the classifier at the given neural message passing step
        """

        # Get initial edge embeddings to
        dist_reid = cosine_distance(track_app, current_app)
        pos_edge_feats = self.compute_edge_feats(
            track_coords, current_coords, track_t, curr_t
        )
        edge_feats = torch.cat((pos_edge_feats, dist_reid.unsqueeze(-1)), dim=-1)
        edge_embeds = self.edge_in_mlp(edge_feats)
        initial_edge_embeds = edge_embeds.clone()

        # Get initial node embeddings, reduce dimensionality from 512 to node_dim
        track_embeds = F.relu(self.cnn_linear(track_app))
        curr_embeds = F.relu(self.cnn_linear(current_app))

        classified_edges = []
        for _ in range(self.num_steps):
            edge_embeds = torch.cat((edge_embeds, initial_edge_embeds), dim=-1)
            edge_embeds, track_embeds, curr_embeds = self.graph_net(
                edge_embeds=edge_embeds,
                nodes_a_embeds=track_embeds,
                nodes_b_embeds=curr_embeds,
            )

            classified_edges.append(self.classifier(edge_embeds))

        return torch.stack(classified_edges).squeeze(-1)
