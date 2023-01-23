"""AFA global config."""
from argparse import Namespace

cfg = {}


def init_config(args: Namespace) -> None:
    """Init global config."""
    cfg["inplace_abn"] = args.inplace_abn
