import argparse
import json
from pathlib import Path

from tqdm import tqdm
import json
import multiprocessing as mp
import psutil

from proteinmpnn.io import parse_pdb_to_dict


def worker(pdb_path: Path) -> str:
    return json.dumps(parse_pdb_to_dict(pdb_path))


def main(args):
    pdb_paths = list(Path(args.input_path).iterdir())
    with (
        mp.Pool(processes=psutil.cpu_count(logical=False)) as pool,
        open(args.output_path, "w") as output_file,
    ):
        result = tqdm(pool.imap(worker, pdb_paths), total=len(pdb_paths))
        for json_line in result:
            output_file.write(f"{json_line}\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--input_path",
        type=str,
        help="Path to a folder with pdb files, e.g. /home/my_pdbs/",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        help="Path where to save .jsonl dictionary of parsed pdbs",
    )
    args = argparser.parse_args()
    main(args)
