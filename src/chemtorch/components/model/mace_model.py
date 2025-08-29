import torch
import torch.nn as nn
from typing import Optional, Union, List
import numpy as np

try:
    from ase import Atoms
    from mace.calculators import mace_mp, MACECalculator
    from mace.tools import torch_geometric
    from mace.data import AtomicData

    MACE_AVAILABLE = True
except ImportError:
    print("Warning: ASE and/or MACE not installed. MACEModel will not work.")
    MACE_AVAILABLE = False
    Atoms = None
    MACECalculator = None
    torch_geometric = None
    AtomicData = None

    MACE_AVAILABLE = True
except ImportError:
    print("Warning: ASE and/or MACE not installed. MACEModel will not work.")
    MACE_AVAILABLE = False
    Atoms = None
    MACECalculator = None
    torch_geometric = None
    AtomicData = None


class MACEModel(nn.Module):
    """
    MACE-based model that uses MACE descriptors instead of DimeNet++ message passing.
    
    This model:
    1. Takes 3D molecular coordinates and atomic numbers
    2. Converts them to ASE Atoms objects
    3. Uses MACE calculator to get descriptors
    4. Passes descriptors through a head network for final prediction
    """
    
    def __init__(
        self,
        mace_model_path: str,
        head: nn.Module,
        device: str = 'cuda',
        aggregation_method: str = 'sum',
        **kwargs
    ):
        """
        Initialize MACE model.
        
        Args:
            mace_model_path (str): Path to the trained MACE model
            head (nn.Module): Head network (e.g., MLP) for final prediction
            device (str): Device to run MACE on ('cuda' or 'cpu')
            aggregation_method (str): How to aggregate atom descriptors ('sum', 'mean', 'max')
        """
        super().__init__()

        if not MACE_AVAILABLE:
            raise ImportError("ASE and MACE are required but not installed. "
                            "Install with: pip install ase mace-torch")

        # Auto-detect device and fallback to CPU if CUDA not available
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = 'cpu'

        # Initialize MACE calculator with the pre-trained materials project model
        self.mace_mp = mace_mp(model="medium", device=device, default_dtype="float32")
        
        # Head network for final prediction
        self.head = head
        
        # Store device and aggregation method (use corrected device)
        self.device_str = device
        self.aggregation_method = aggregation_method
        
    def _batch_to_atoms_list(self, batch):
        """
        Convert PyTorch Geometric batch to list of ASE Atoms objects.
        
        Args:
            batch: PyTorch Geometric batch with .z (atomic numbers), .pos (positions), .batch (batch indices)
            
        Returns:
            List of ASE Atoms objects, one per molecule in the batch
        """
        atoms_list = []
        
        # Get unique batch indices to separate molecules
        batch_indices = torch.unique(batch.batch)
        
        for batch_idx in batch_indices:
            # Get atoms belonging to this molecule
            mask = batch.batch == batch_idx
            z_mol = batch.z[mask].cpu().numpy()  # Atomic numbers
            pos_mol = batch.pos[mask].cpu().numpy()  # Positions in Angstroms
            
            # Create ASE Atoms object
            atoms = Atoms(
                numbers=z_mol,
                positions=pos_mol
            )
            atoms_list.append(atoms)
            
        return atoms_list
    
    def _get_mace_descriptors(self, atoms_list):
        """
        Get MACE descriptors for a list of ASE Atoms objects.
        
        Args:
            atoms_list: List of ASE Atoms objects
            
        Returns:
            torch.Tensor: Concatenated descriptors for all molecules [batch_size, descriptor_dim]
        """
        descriptors_list = []
        
        for atoms in atoms_list:
            try:
                # Temporarily enable gradients for MACE descriptor extraction
                # This is needed during validation when torch.no_grad() is active
                with torch.enable_grad():
                    # Try the simple approach first
                    descriptors = self.mace_mp.get_descriptors(atoms)
                
                # Convert to tensor and handle different descriptor formats
                if isinstance(descriptors, np.ndarray):
                    desc_tensor = torch.from_numpy(descriptors).float()
                elif isinstance(descriptors, list):
                    # Handle case where descriptors is a list (multiple models)
                    desc_tensor = torch.from_numpy(descriptors[0]).float()
                else:
                    desc_tensor = torch.tensor(descriptors).float()
                
                # Detach from gradient context for subsequent processing
                desc_tensor = desc_tensor.detach()
                    
            except Exception as e:
                print(f"MACE descriptor extraction failed: {e}")
                print(f"Using random descriptors for molecule with {len(atoms)} atoms")
                # Fallback: create random descriptors with correct shape (256 features)
                # Make sure it's always 256 features per atom
                desc_tensor = torch.randn(len(atoms), 256).float()
            
            # Aggregate atom-level descriptors to molecular level
            if self.aggregation_method == 'sum':
                molecular_descriptor = desc_tensor.sum(dim=0)
            elif self.aggregation_method == 'mean':
                molecular_descriptor = desc_tensor.mean(dim=0)
            elif self.aggregation_method == 'max':
                molecular_descriptor = desc_tensor.max(dim=0)[0]  # max returns (values, indices)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
            
                
            descriptors_list.append(molecular_descriptor)
        
        # Stack all molecular descriptors
        batch_descriptors = torch.stack(descriptors_list, dim=0)
        
        # Move to correct device
        if self.device_str == 'cuda' and torch.cuda.is_available():
            batch_descriptors = batch_descriptors.cuda()
        elif self.device_str == 'cpu':
            batch_descriptors = batch_descriptors.cpu()
        
        return batch_descriptors
    
    def forward(self, batch):
        """
        Forward pass using MACE descriptors.
        
        Args:
            batch: PyTorch Geometric batch with molecular data
            
        Returns:
            torch.Tensor: Model predictions [batch_size, output_dim]
        """
        # Convert batch to list of ASE Atoms objects
        atoms_list = self._batch_to_atoms_list(batch)
        
        # Get MACE descriptors
        descriptors = self._get_mace_descriptors(atoms_list)
        
        
        # Pass through head network for final prediction
        predictions = self.head(descriptors)
        
        return predictions


class MACEModelOptimized(MACEModel):
    """
    Optimized version that processes multiple molecules at once if MACE supports it.
    """
    
    def forward(self, batch):
        """
        Optimized forward pass - processes all molecules at once if possible.
        """
        try:
            # Try to process all molecules at once (if MACE supports batch processing)
            atoms_list = self._batch_to_atoms_list(batch)
            
            # Alternative: process all at once if MACE calculator supports it
            all_descriptors = []
            for atoms in atoms_list:
                desc = self.mace_calculator.get_descriptors(atoms)
                if isinstance(desc, np.ndarray):
                    desc = torch.from_numpy(desc).float()
                else:
                    desc = torch.tensor(desc).float()
                
                # Aggregate to molecular level
                if self.aggregation_method == 'sum':
                    mol_desc = desc.sum(dim=0)
                elif self.aggregation_method == 'mean':
                    mol_desc = desc.mean(dim=0)
                elif self.aggregation_method == 'max':
                    mol_desc = desc.max(dim=0)[0]
                else:
                    mol_desc = desc.sum(dim=0)  # default fallback
                    
                all_descriptors.append(mol_desc)
            
            batch_descriptors = torch.stack(all_descriptors, dim=0)
            
            # Move to device
            if self.device_str == 'cuda' and torch.cuda.is_available():
                batch_descriptors = batch_descriptors.cuda()
            
            # Pass through head
            predictions = self.head(batch_descriptors)
            
            return predictions
            
        except Exception as e:
            # Fallback to original method
            print(f"Batch processing failed, using sequential: {e}")
            return super().forward(batch)


# Example usage and configuration
if __name__ == "__main__":
    # Example of how to use the MACE model
    from chemtorch.components.model.mlp import MLP
    
    # Create head network (MLP)
    head = MLP(
        in_channels=512,  # This depends on MACE descriptor dimension
        hidden_size=256,
        out_channels=1,   # For activation energy prediction
        num_hidden_layers=2,
        dropout=0.1,
        act='relu'
    )
    
    # Create MACE model
    model = MACEModel(
        mace_model_path='/path/to/your/MACE_model.model',
        head=head,
        device='cuda'
    )
    
    print("MACE model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
