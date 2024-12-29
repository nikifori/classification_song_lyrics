'''
@File    :   training.py
@Time    :   12/2024
@Author  :   nikifori
@Version :   -
'''
import lightning as L
from hyperpyyaml import load_hyperpyyaml
import argparse
from pathlib import Path
import os


def safe_load_yaml(yaml_file):
    """
    Safely loads a yaml file, using hyperpyyaml.
    Args:
        yaml_file (str): The path to the yaml file.
    Returns:
        dict: The dictionary containing the yaml file data.
    """
    with open(yaml_file, "r") as f:
        cfg_dict = load_hyperpyyaml(f)
    return cfg_dict

def train(cfg):
    ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="YAML file containing the training configuration.",
    )
    parser.add_argument("-g", "--gpu", type=str, default="0", help="Using gpu #")
    args = parser.parse_args()

    hparams_file = Path(args.config).resolve()
    print(f"config file {hparams_file}")

    cfg = safe_load_yaml(args.config)
    cfg["config_filename"] = hparams_file

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train(cfg)