"""Utility script to generate config files from a base config and a defined set of variations"""
import argparse
import ast
import itertools
import sys
from pathlib import Path

from experiments.utils.config import read_config, dump_config

parser = argparse.ArgumentParser(description="train on LSTM addition task")
default_config = Path(__file__).absolute().parent.parent / 'experiments' / 'config.yml.example'
parser.add_argument("--base_config", type=Path, default=default_config, help="path to configuration file")
parser.add_argument("--change",
                    action='append',
                    nargs=2,
                    type=str,
                    metavar=("KEY", "VALUE"),
                    default=[],
                    help="change value for given key")
parser.add_argument("--seeds", nargs=2, type=int, metavar=("START", "STOP"), help="beginning and end of seed range")
args = parser.parse_args()

codebase_path = Path(__file__).absolute().parent.parent.parent
sys.path.append(str(codebase_path))

# Place to store the configs
out_dir = Path(__file__).parent / "generated_configs"
out_dir.mkdir()

base_config = read_config(args.base_config)

# Dictionary of keys to modify. Variations have to be added as a list of options
modify_key = {}
for k, v in args.change:
    modify_key.setdefault(k, []).append(type(base_config.get(k, ""))(v))
if args.seeds is not None:
    modify_key['global_seed'] = range(args.seeds[0], args.seeds[1] + 1)
print(modify_key)
option_names = list(modify_key.keys())

for i, options in enumerate(itertools.product(*[val for val in modify_key.values()])):

    for key, val in zip(option_names, options):
        base_config[key] = val

    name_parts = [f"{key}{val}" if isinstance(val, int) else str(val) for key, val in zip(option_names, options)]

    base_config["experiment_name"] = "_".join(name_parts)

    dump_config(base_config, out_dir, f"config_{i+1}.yml")
