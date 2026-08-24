from rdkit import Chem
import torch


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

from torch_geometric.data import Data


def bond_features(bond: Chem.Bond) -> list[float]:
    """Convert an RDKit bond into a numerical feature vector."""
    bond_type = bond.GetBondType()

    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
    ]

def mol_to_graph(smiles: str) -> Data:
    """Convert a SMILES string into a PyTorch Geometric molecular graph."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Atom feature matrix
    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    # Bond connectivity
    edges = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # Molecular graphs are represented as directed edges
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