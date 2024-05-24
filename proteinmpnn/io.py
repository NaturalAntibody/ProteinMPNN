from Bio.PDB import PDBParser
from Bio.SeqUtils import IUPACData
from pathlib import Path
from typing import Any, Optional


def parse_pdb_to_dict(pdb_path: Path, chain_ids: Optional[list[str]] = None) -> dict[str, Any]:
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("-", pdb_path)
    chain_seqs = {}
    chain_coords = {}

    for chain in structure.get_chains():
        if chain.id not in chain_ids:
            continue
        coords_N = []
        coords_O = []
        coords_CA = []
        coords_C = []
        seq = []
        for residue in chain:
            seq.append(
                IUPACData.protein_letters_3to1[residue.get_resname().capitalize()]
            )
            for atom in residue.get_atoms():
                coords = atom.get_coord().tolist()
                match atom.id:
                    case "N":
                        coords_N.append(coords)
                    case "O":
                        coords_O.append(coords)
                    case "CA":
                        coords_CA.append(coords)
                    case "C":
                        coords_C.append(coords)
        chain_coords[f"coords_chain_{chain.id}"] = {
            f"N_chain_{chain.id}": coords_N,
            f"O_chain_{chain.id}": coords_O,
            f"C_chain_{chain.id}": coords_C,
            f"CA_chain_{chain.id}": coords_CA,
        }
        chain_seqs[f"seq_chain_{chain.id}"] = "".join(seq)

    return {
        **chain_seqs,
        **chain_coords,
        "seq": "".join(chain_seqs.values()),
        "name": pdb_path.stem,
        "num_of_chains": len(chain_seqs),
    }