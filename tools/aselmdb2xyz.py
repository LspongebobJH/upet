#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from glob import glob
from pathlib import Path

from ase.db import connect
from ase.io import write
from tqdm import tqdm


def resolve_aselmdb_paths(input_path: str) -> list[Path]:
    path = Path(input_path)

    if path.is_file():
        db_paths = [path]
    elif path.is_dir():
        db_paths = sorted(path.glob("*.aselmdb"))
    else:
        db_paths = [Path(p) for p in sorted(glob(input_path))]

    if not db_paths:
        raise FileNotFoundError(f"No ASELMDB files found for path: {input_path}")

    if any("part_" in db_path.name for db_path in db_paths):
        db_paths = sorted(db_paths, key=part_sort_key)

    return [db_path.resolve() for db_path in db_paths]


def part_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"part_(\d+)", path.name)
    if match is None:
        return (-1, path.name)
    return (int(match.group(1)), path.name)


def row_to_atoms(row):
    try:
        return row.toatoms(add_additional_information=True)
    except TypeError:
        return row.toatoms()


def convert_aselmdb_to_xyz(input_path: str, output_path: str, force: bool = False) -> int:
    db_paths = resolve_aselmdb_paths(input_path)
    output = Path(output_path).resolve()

    if output.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {output}. Use --force to overwrite it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    total_structures = 0
    append = False

    for db_path in db_paths:
        with connect(db_path, readonly=True, use_lock_file=False) as database:
            for row in tqdm(database.select(), total=database.count(), desc=f"Converting {db_path.name}"):
                atoms = row_to_atoms(row)
                write(output, atoms, format="extxyz", append=append)
                append = True
                total_structures += 1

    return total_structures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an ASELMDB dataset into a single extended XYZ file."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="ASELMDB file, directory containing *.aselmdb files, or glob pattern.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output .xyz path. The file is written in extxyz format.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    num_structures = convert_aselmdb_to_xyz(
        input_path=args.input_path,
        output_path=args.output_path,
        force=args.force,
    )
    print(f"Wrote {num_structures} structures to {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
