from rdkit import Chem
import torch
from torch_geometric.data import Data


def atom_features(atom: Chem.Atom) -> list[float]:
    """Convert an RDKit atom into a numerical feature vector."""
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetTotalDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()),
        float(atom.GetHybridization()),
    ]


def mol_to_graph(smiles: str) -> Data:
    """Convert a SMILES string into a PyTorch Geometric molecular graph."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    edges = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges.append([i, j])
        edges.append([j, i])

    if edges:
        edge_index = torch.tensor(
            edges,
            dtype=torch.long,
        ).t().contiguous()
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    return Data(
        x=x,
        edge_index=edge_index,
    )
