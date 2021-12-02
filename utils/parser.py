import sys
import argparse
import yaml


class AttributeDict(dict):
    """
    A modified version of dictionary that allows its keys to be accessed as attributes of the dictionary.


    """
    def __init__(self, d: dict):
        for k, v in d.items():
            if type(v) == dict:
                d[k] = AttributeDict(v)
        self.__dict__ = d


def parse_args():
    parser = argparse.ArgumentParser(
        description=("JEPS memory implementations.")
    )

    parser.add_argument(
        "--cfg",
        dest="cfg_path",
        help="Path to the config file",
        default="configs/default.yaml",
        type=str,
    )

    return parser.parse_args()


def load_config(config_file_path):
    """
    
    Parameters
    ----------
    config_file_path : str
        The path to the configuration file to be loaded

    Returns
    -------
    A utils.parser.AttributeDict object with the settings accessible as attributes of this object.
    """
    with open(config_file_path) as cfg_file:
        cfg_dict = yaml.safe_load(cfg_file)

    return AttributeDict(cfg_dict)


if __name__ == '__main__':
    cfg_dict = load_config('/home/abrsh/Ego4D/Ego4Dv1/Ego4D_keyframe_localisation/configs/vivit_config.yaml')
