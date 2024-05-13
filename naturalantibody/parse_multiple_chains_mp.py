import argparse
import json
from pathlib import Path

from tqdm import tqdm
from research.config import DATA_PATH
from Bio.PDB import PDBParser
from Bio.SeqUtils import IUPACData
import json
import multiprocessing as mp
import psutil

pdb_parser = None

def initializer():
    global pdb_parser
    pdb_parser = PDBParser()


def parse_pdb_to_json_str(pdb_path: Path) -> str:
    structure = pdb_parser.get_structure("-", pdb_path)
    chain_seqs = {}
    chain_coords = {}

    for chain in structure.get_chains():
        coords_N = []
        coords_O = []
        coords_CA = []
        coords_C = []
        seq = []
        for residue in chain:
            seq.append(IUPACData.protein_letters_3to1[
                residue.get_resname().capitalize()])
            for atom in residue.get_atoms():
                coords = atom.get_coord().tolist()
                match atom.id:
                    case 'N':
                        coords_N.append(coords)
                    case 'O':
                        coords_O.append(coords)
                    case 'CA':
                        coords_CA.append(coords)
                    case 'C':
                        coords_C.append(coords)
        chain_coords[f"coords_chain_{chain.id}"] = {
            f"N_chain_{chain.id}": coords_N,
            f"O_chain_{chain.id}": coords_O,
            f"C_chain_{chain.id}": coords_C,
            f"CA_chain_{chain.id}": coords_CA
        }
        chain_seqs[f"seq_chain_{chain.id}"] = "".join(seq)

    result = {
        **chain_seqs,
        **chain_coords,
        "seq": "".join(chain_seqs.values()),
        "name": pdb_path.stem,
        "num_of_chains": len(chain_seqs)
    }
    return json.dumps(result)


def main(args):
    pdb_paths = list(Path(args.input_path).iterdir())
    with mp.Pool(initializer=initializer, processes=psutil.cpu_count(logical=False)) as pool, open(args.output_path, "w") as output_file:
        result = tqdm(pool.imap(parse_pdb_to_json_str, pdb_paths), total=len(pdb_paths))
        for json_line in result:
            output_file.write(f"{json_line}\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, help="Path to a folder with pdb files, e.g. /home/my_pdbs/")
    argparser.add_argument("--output_path", type=str, help="Path where to save .jsonl dictionary of parsed pdbs")
    args = argparser.parse_args()
    main(args)
