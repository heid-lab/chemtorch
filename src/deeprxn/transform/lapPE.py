import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
)

from deeprxn.data_pipeline.data_pipeline import DataPipelineComponent


class LapPE(DataPipelineComponent):
    """
    # TODO: check out how to cite code
    """

    def __init__(
        self,
        eigvec_norm,
        max_freqs,
        attr_name,
        is_undirected=False,
        laplacian_norm_type=None,
        type: str = "graph",
    ) -> None:
        self.eigvec_norm = eigvec_norm
        self.max_freqs = max_freqs
        self.laplacian_norm_type = laplacian_norm_type
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:

        N = data.x.shape[0]

        undir_edge_index = to_undirected(data.edge_index)

        # Eigen values and vectors.
        evals, evects = None, None

        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(
                undir_edge_index,
                normalization=self.laplacian_norm_type,
                num_nodes=N,
            )
        )
        evals, evects = np.linalg.eigh(L.toarray())

        data.EigVals, data.EigVecs = self.get_lap_decomp_stats(
            evals=evals,
            evects=evects,
            max_freqs=self.max_freqs,
            eigvec_norm=self.eigvec_norm,
        )

        return data

    def get_lap_decomp_stats(self, evals, evects, max_freqs, eigvec_norm="L2"):
        """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

        Args:
            evals, evects: Precomputed eigen-decomposition
            max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
            eigvec_norm: Normalization for the eigen vectors of the Laplacian
        Returns:
            Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
            Tensor (num_nodes, max_freqs) of eigenvector values per node
        """
        N = len(evals)  # Number of nodes, including disconnected nodes.

        # Keep up to the maximum desired number of frequencies.
        idx = evals.argsort()[:max_freqs]
        evals, evects = evals[idx], np.real(evects[:, idx])
        evals = torch.from_numpy(np.real(evals)).clamp_min(0)

        # Normalize and pad eigen vectors.
        evects = torch.from_numpy(evects).float()
        evects = self.eigvec_normalizer(
            evects, evals, normalization=eigvec_norm
        )
        if N < max_freqs:
            EigVecs = F.pad(evects, (0, max_freqs - N), value=float("nan"))
        else:
            EigVecs = evects

        # Pad and save eigenvalues.
        if N < max_freqs:
            EigVals = F.pad(
                evals, (0, max_freqs - N), value=float("nan")
            ).unsqueeze(0)
        else:
            EigVals = evals.unsqueeze(0)
        EigVals = EigVals.repeat(N, 1).unsqueeze(2)

        # print(f"Eigenvalues: {EigVals.shape}")
        # print(EigVals)
        # print(f"Eigenvectors: {EigVecs.shape}")
        # print(EigVecs)
        # lol = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2)
        # print(f"Concatenated: {lol.shape}")
        # print(lol)
        # assert False

        return EigVals, EigVecs

    def eigvec_normalizer(
        self, EigVecs, EigVals, normalization="L2", eps=1e-12
    ):
        """
        Implement different eigenvector normalizations.
        """

        EigVals = EigVals.unsqueeze(0)

        if normalization == "L1":
            # L1 normalization: eigvec / sum(abs(eigvec))
            denom = EigVecs.norm(p=1, dim=0, keepdim=True)

        elif normalization == "L2":
            # L2 normalization: eigvec / sqrt(sum(eigvec^2))
            denom = EigVecs.norm(p=2, dim=0, keepdim=True)

        elif normalization == "abs-max":
            # AbsMax normalization: eigvec / max|eigvec|
            denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

        elif normalization == "wavelength":
            # AbsMax normalization, followed by wavelength multiplication:
            # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
            denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
            eigval_denom = torch.sqrt(EigVals)
            eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
            denom = denom * eigval_denom * 2 / np.pi

        elif normalization == "wavelength-asin":
            # AbsMax normalization, followed by arcsin and wavelength multiplication:
            # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
            denom_temp = (
                torch.max(EigVecs.abs(), dim=0, keepdim=True)
                .values.clamp_min(eps)
                .expand_as(EigVecs)
            )
            EigVecs = torch.asin(EigVecs / denom_temp)
            eigval_denom = torch.sqrt(EigVals)
            eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
            denom = eigval_denom

        elif normalization == "wavelength-soft":
            # AbsSoftmax normalization, followed by wavelength multiplication:
            # eigvec / (softmax|eigvec| * sqrt(eigval))
            denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(
                dim=0, keepdim=True
            )
            eigval_denom = torch.sqrt(EigVals)
            eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
            denom = denom * eigval_denom

        else:
            raise ValueError(f"Unsupported normalization `{normalization}`")

        denom = denom.clamp_min(eps).expand_as(EigVecs)
        EigVecs = EigVecs / denom

        return EigVecs
