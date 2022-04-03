import argparse

import gensim

from config.qa_default import get_cfg_defaults
from runners import QARunner


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--use-wandb', action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config is not None:
        cfg.merge_from_file("config/" + args.config + ".yaml")
    cfg.freeze()
    runner = QARunner(cfg, args.use_wandb)
    runner.train()
