import argparse

import gensim

from config.default import get_cfg_defaults
from runners import LocalizationRunner


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--use-wandb', action="store_true", default=False)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config is not None:
        cfg.merge_from_file("config/" + args.config + ".yaml")
    cfg.freeze()
    runner = LocalizationRunner(cfg, args.use_wandb)
    runner._load_model(args.checkpoint)
    runner.qualitative_eval(["moment_retrieval"], [{"top_n": 5, "min_thresh": 0.0, "max_thresh": 0.01,
                                                    "min_start": 16, "max_end": 112, "min_length": 16,
                                                    "max_length": 120, "by_frame": False, "max_number": 50,
                                                    "display_interval": 50, "filename": args.filename}])
