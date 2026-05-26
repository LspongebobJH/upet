from glob import glob
from pathlib import Path
import re
import datetime
from pathlib import Path
import pandas as pd

import pickle
from tqdm import tqdm
from ase.db import connect
from ase.io import read

def resolve_aselmdb_paths(valid_data_path):
    db_paths = sorted(glob(valid_data_path))
    if not db_paths and Path(valid_data_path).is_file():
        db_paths = [valid_data_path]
    if not db_paths:
        raise FileNotFoundError(f"No ASELMDB files found for path: {valid_data_path}")
    if any("part_" in Path(path).name for path in db_paths):
        db_paths = sorted(
            db_paths,
            key=lambda path: int(re.search(r"part_(\d+)", Path(path).name).group(1))
            if re.search(r"part_(\d+)", Path(path).name)
            else Path(path).name,
        )
    return db_paths

def resolve_xyz_paths(valid_data_path):
    p = Path(valid_data_path)
    if p.is_file() and p.suffix.lower() == ".xyz":
        return [str(p.resolve())]
    paths = sorted(glob(valid_data_path))
    xyz_paths = [x for x in paths if str(x).lower().endswith(".xyz")]
    if xyz_paths:
        return xyz_paths
    if p.is_dir():
        xyz_paths = sorted(glob(str(p / "*.xyz")))
        if xyz_paths:
            return xyz_paths
    return []

def load_eval_data(valid_data_path):
    if valid_data_path.endswith(".pkl"):
        print(f"Detected pickle input: {valid_data_path}")
        with open(valid_data_path, "rb") as f:
            return pickle.load(f)

    if valid_data_path.endswith(".xyz"):
        xyz_paths = resolve_xyz_paths(valid_data_path)
        print(f"Detected XYZ input with {len(xyz_paths)} file(s)")
        eval_data = []
        for xyz_path in tqdm(xyz_paths, desc="Loading XYZ files"):
            atoms_list = read(xyz_path, index=":")
            if not isinstance(atoms_list, list):
                atoms_list = [atoms_list]
            eval_data.extend(atoms_list)
        print(f"Total structures in concatenated atomlist: {len(eval_data)}")
        return eval_data

    if ".aselmdb" in valid_data_path:
        db_paths = resolve_aselmdb_paths(valid_data_path)
        print(f"Detected ASELMDB input with {len(db_paths)} file(s)")
        eval_data = []
        for db_path in tqdm(db_paths, desc="Loading ASELMDB files"):
            with connect(db_path, readonly=True, use_lock_file=False) as database:
                eval_data.extend(row.toatoms() for row in database.select())
        return eval_data

    raise ValueError(
        f"Unsupported evaluation data format for path: {valid_data_path}. "
        "Expected a .pkl file, .xyz path/glob/directory, or .aselmdb path/glob."
    )
class Logger:
    def __init__(self, log_path: str):
        assert log_path is not None, "Logger requires a valid log_path"
        self.log_path = log_path
        self.df_path = Path(log_path).with_suffix(".csv")
        self.f = self.open()

    def open(self):
        log_path = Path(self.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return open(log_path, "a", encoding="utf-8")

    def __call__(self, msg: str):
        print(msg)
        self.f.write(datetime.datetime.now().isoformat(timespec="minutes") + " - " + msg + "\n")
        self.f.flush()

    def log_df(self, df: pd.DataFrame):
        df.to_csv(self.df_path, index=False)

    def close(self):
        self.f.close()