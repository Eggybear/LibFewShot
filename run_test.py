# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import argparse
import copy
import os
import torch
from core.config import Config
from core import Test


PATH = "./results/LDPNet-miniImageNet--ravi-resnet10-5-5-Jun-02-2026-12-41-35"
VAR_DICT = {
    "data_root": "./data/StanfordCar",
    "image_size": 224,
    "test_epoch": 1,
    "device_ids": "0",
    "n_gpu": 1,
    "seed": 1111,
    "deterministic": True,
    "test_way": 5,
    "test_shot": 5,
    "test_query": 15,
    "test_episode": 600,
    "episode_size": 1,
    "workers": 4,
    "pin_memory": False,
    "test_checkpoint": "model_last.pth",
    "classifier": {
        "name": "LDPNet",
        "kwargs": {
            "lamba1": 1.0,
            "lamba2": 0.15,
            "momentum": 0.998,
            "beta": 0.5,
            "query_aug_times": 1,
            "use_lr_classifier": True,
            "transductive": True,
            "transductive_iter": 7,
            "transductive_k": 10,
            "lr_max_iter": 1000,
            "local_query_crops": 6,
        },
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run LDPNet evaluation.")
    parser.add_argument("--result_path", default=PATH, help="Training result directory.")
    parser.add_argument(
        "--data_root", default=VAR_DICT["data_root"], help="Target dataset root."
    )
    parser.add_argument("--test_shot", type=int, default=VAR_DICT["test_shot"])
    parser.add_argument("--test_episode", type=int, default=VAR_DICT["test_episode"])
    parser.add_argument("--test_epoch", type=int, default=VAR_DICT["test_epoch"])
    parser.add_argument("--device_ids", default=VAR_DICT["device_ids"])
    parser.add_argument("--n_gpu", type=int, default=VAR_DICT["n_gpu"])
    parser.add_argument("--checkpoint", default=VAR_DICT["test_checkpoint"])
    parser.add_argument(
        "--transductive",
        action="store_true",
        default=None,
        help="Use the author's full-data transductive refinement.",
    )
    return parser.parse_args()


def main(rank, config, result_path):
    test = Test(rank, config, result_path)
    test.test_loop()


if __name__ == "__main__":
    args = parse_args()
    var_dict = copy.deepcopy(VAR_DICT)
    var_dict.update(
        {
            "data_root": args.data_root,
            "test_shot": args.test_shot,
            "test_episode": args.test_episode,
            "test_epoch": args.test_epoch,
            "device_ids": args.device_ids,
            "n_gpu": args.n_gpu,
            "test_checkpoint": args.checkpoint,
        }
    )
    if args.transductive is not None:
        var_dict["classifier"]["kwargs"]["transductive"] = args.transductive

    config = Config(
        os.path.join(args.result_path, "config.yaml"), var_dict
    ).get_config_dict()

    if config["n_gpu"] > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["device_ids"]
        torch.multiprocessing.spawn(
            main, nprocs=config["n_gpu"], args=(config, args.result_path)
        )
    else:
        main(0, config, args.result_path)
