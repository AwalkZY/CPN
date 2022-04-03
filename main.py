import argparse

import gensim

from config.default import get_cfg_defaults
from runners import LocalizationRunner


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
    runner = LocalizationRunner(cfg, args.use_wandb)
    runner._load_model("/home1/zhaoyang/codes/CVPR2021/best_cvpr/localization/anet2.pt")
    # runner.train()
    runner.eval(0, ["moment_retrieval"], [{"top_n": 5, "thresh": 0.5, "by_frame": False, "display_interval": 100}])