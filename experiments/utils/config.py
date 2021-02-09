from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML


def read_config(cfg_path: Path) -> dict:
    if cfg_path.exists():
        with cfg_path.open('r') as fp:
            yaml = YAML(typ="safe")
            cfg = yaml.load(fp)
    else:
        raise FileNotFoundError(cfg_path)

    cfg = parse_config(cfg)

    return cfg


def dump_config(cfg: dict, folder: Path, filename: str = 'config.yml'):
    cfg_path = folder / filename
    if not cfg_path.exists():
        with cfg_path.open('w') as fp:
            temp_cfg = {}
            for key, val in cfg.items():
                if any([x in key for x in ['dir', 'path', 'file']]):
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            temp_list.append(str(element))
                        temp_cfg[key] = temp_list
                    else:
                        temp_cfg[key] = str(val)
                elif isinstance(val, pd.Timestamp):
                    temp_cfg[key] = val.strftime(format="%d/%m/%Y")
                else:
                    temp_cfg[key] = val

            yaml = YAML()
            yaml.dump(dict(OrderedDict(sorted(temp_cfg.items()))), fp)
    else:
        FileExistsError(cfg_path)


def parse_config(cfg: dict) -> dict:

    for key, val in cfg.items():
        # convert all path strings to PosixPath objects
        if any([x in key for x in ['dir', 'path', 'file']]):
            if (val is not None) and (val != "None"):
                if isinstance(val, list):
                    temp_list = []
                    for element in val:
                        temp_list.append(Path(element))
                    cfg[key] = temp_list
                else:
                    cfg[key] = Path(val)
            else:
                cfg[key] = None

        # convert Dates to pandas Datetime indexs
        elif key.endswith('_date'):
            cfg[key] = pd.to_datetime(val, format='%d/%m/%Y')

        elif any(key == x for x in ["static_inputs", "camels_attributes"]):
            if val is None:
                cfg[key] = []
        else:
            pass

    return cfg
