import json
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import IUPACData
from pathlib import Path
from typing import Any, Optional, TextIO

import torch

Structure = dict[str, Any]

def parse_pdb(pdb: Path | TextIO, chain_ids: Optional[list[str]] = None) -> Structure:  # noqa: F821

    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure("-", pdb)
    chain_seqs = {}
    chain_coords = {}

    assert structure is not None, "Failed to parse PDB file."
    for chain in structure.get_chains():
        if chain_ids and chain.id not in chain_ids:
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
        "name": pdb.name,
        "num_of_chains": len(chain_seqs),
    }


def select_chains(protein: dict, chains: list[str]) -> dict:
    res = {"name": protein["name"], "num_of_chains": len(chains), "seq": ""}
    for chain in chains:
        res[f"coords_chain_{chain}"] = protein[f"coords_chain_{chain}"]
        res[f"seq_chain_{chain}"] = protein[f"seq_chain_{chain}"]
        res["seq"] += protein[f"seq_chain_{chain}"]
    return res


def write_scores(id: str, designed_score: torch.Tensor, global_score: torch.Tensor, out_jsonl: TextIO):
    out_json = {
        "id": "pdb",
        "scores": designed_score.tolist(),
        "global_scores": global_score.tolist(),
    }
    out_jsonl.write(f"{json.dumps(out_json)}\n")


if __name__ == "__main__":

    pdb_path = Path("1ahw_modelled.pdb")
    chain_ids = ["H", "L"]  # Example chain IDs to select

    with open(pdb_path, "r") as pdb_file:
        print(f"{pdb_file.name=}")
        print(f"{type(pdb_file)=}")
        protein = parse_pdb(pdb_file, chain_ids=chain_ids)
        print(json.dumps(protein, indent=2))