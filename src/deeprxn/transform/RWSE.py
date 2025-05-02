import torch
from torch_geometric.data import Data
from torch_geometric.utils import scatter, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes

from deeprxn.data_pipeline.data_pipeline import DataPipelineComponent


class RWSE(DataPipelineComponent):
    """
    # TODO: check out how to cite code
    """

    def __init__(
        self,
        times: int = 20,
        type: str = "graph",
    ) -> None:
        self.times = times

    def forward(self, data: Data) -> Data:

        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.

        times = list(range(1, self.times + 1))
        rw_landing = self.get_rw_landing_probs(
            ksteps=times, edge_index=data.edge_index, num_nodes=N
        )
        data.wandomwalkpe = rw_landing

        return data

    def get_rw_landing_probs(
        self, ksteps, edge_index, edge_weight=None, num_nodes=None, space_dim=0
    ):
        """Compute Random Walk landing probabilities for given list of K steps.

        Args:
            ksteps: List of k-steps for which to compute the RW landings
            edge_index: PyG sparse representation of the graph
            edge_weight: (optional) Edge weights
            num_nodes: (optional) Number of nodes in the graph
            space_dim: (optional) Estimated dimensionality of the space. Used to
                correct the random-walk diagonal by a factor `k^(space_dim/2)`.
                In euclidean space, this correction means that the height of
                the gaussian distribution stays almost constant across the number of
                steps, if `space_dim` is the dimension of the euclidean space.

        Returns:
            2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
        """
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1), device=edge_index.device
            )
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        source, dest = edge_index[0], edge_index[1]
        deg = scatter(
            edge_weight, source, dim=0, dim_size=num_nodes, reduce="sum"
        )  # Out degrees.
        deg_inv = deg.pow(-1.0)
        deg_inv.masked_fill_(deg_inv == float("inf"), 0)

        if edge_index.numel() == 0:
            P = edge_index.new_zeros((1, num_nodes, num_nodes))
        else:
            # P = D^-1 * A
            P = torch.diag(deg_inv) @ to_dense_adj(
                edge_index, max_num_nodes=num_nodes
            )  # 1 x (Num nodes) x (Num nodes)
        rws = []
        if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
            # Efficient way if ksteps are a consecutive sequence (most of the time the case)
            Pk = P.clone().detach().matrix_power(min(ksteps))
            for k in range(min(ksteps), max(ksteps) + 1):
                rws.append(
                    torch.diagonal(Pk, dim1=-2, dim2=-1)
                    * (k ** (space_dim / 2))
                )
                Pk = Pk @ P
        else:
            # Explicitly raising P to power k for each k \in ksteps.
            for k in ksteps:
                rws.append(
                    torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1)
                    * (k ** (space_dim / 2))
                )
        rw_landing = torch.cat(rws, dim=0).transpose(
            0, 1
        )  # (Num nodes) x (K steps)

        return rw_landing
