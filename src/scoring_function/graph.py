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

    # Atom features
    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float,
    )

    # Bond connectivity + bond features
    edges = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        features = bond_features(bond)

        # Add both directions because molecular graphs are represented
        # as directed edges in PyTorch Geometric.
        edges.append([i, j])
        edge_features.append(features)

        edges.append([j, i])
        edge_features.append(features)

    if edges:
        edge_index = torch.tensor(
            edges,
            dtype=torch.long,
        ).t().contiguous()

        edge_attr = torch.tensor(
            edge_features,
            dtype=torch.float,
        )
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

        edge_attr = torch.empty(
            (0, 4),
            dtype=torch.float,
        )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )