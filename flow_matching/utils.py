#!/usr/bin/env python
import os
import sys
import time
main_dir = "/home/sheels/Fall2025/10617/DiffuSeq/FlowSeq/"
os.chdir(main_dir)
sys.path.append(main_dir)
sys.path.append(os.path.join(main_dir, "flow_matching"))
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

class RNGKeys:
    def __init__(self):
        self.DataGenerationKey = 123
        self.ModelInitKey = 456
        self.FMInitKey = 789
        self.MainLoopKey = 0


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


class Config:
    def __init__(self, main_dir, time_str):
        default_config = {
            # embedding, not usually changed
            "vocab_size": 30522,
            "embedding_dimension": 128,
            "config_name": "bert-base-uncased",
            "len_dim": 64,
            
            # data
            "single_loop": False,
            # choose from ["unconditional_generation", "seq_to_seq", "seq_to_seq_conditional", "seq_to_seq_conditional_onehot", "seq_to_seq_conditional_onehot_scaled"]
            "mode": "unconditional_generation", 
            # choose from ["train", "valid", "test"]
            "split": "train",
            "use_random_embeddings": False,
            "random_emb_key": 101, # only used for random embeddings
            "scale_value": 5, # only used for seq_to_seq_conditional_onehot_scaled

            #optimizer
            "bsz": 256,
            "num_steps": 1000000,
            "lr": 3e-4,
            "max_grad_norm": 1.,
            "warmup_steps": 10000, # ~ 20 epochs
            "transition_steps": 50000, # ~ 100 epochs
            "decay_rate": 0.9,
            "transition_begin": 200000, # ~ 400 epochs
            
            # checkpointing/evaluation
            "checkpointing_interval": 10000000,
            "print_interval": 1000,

            # model
            "model": "DiTFlowModel",
            "write_model": True,
            "checkpoint_dir": "", # load from checkpoint if not empty
            "model__num_layers": 2,
            "model__num_heads": 4,
            "model__mlp_ratio": 2,
            "model__dropout_rate": 0.,
            "model__hidden_dim": 256,
            "model__time_emb_dim": 256,
            
            # debugging
            "OVERFIT": False,
            "NO_REDIRECT": False,
            "DUMMY": False,
            "MNIST": False
        }
        self.parser = argparse.ArgumentParser()
        add_dict_to_argparser(self.parser, default_config)
        self.args = self.parser.parse_args()
        self.main_dir = main_dir
        self.time_str = time_str
        self.output_dir = os.path.join(self.main_dir, f"diffusion_models/{self.time_str}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.args.DUMMY:
            self.args.embedding_dimension = 2
            self.args.len_dim = 1
            self.args.mode = "unconditional_generation"
        if self.args.MNIST:
            self.args.embedding_dimension = 28
            self.args.len_dim = 28
            self.args.mode = "unconditional_generation"
        
        if "one_hot" in self.args.mode:
            assert self.args.embedding_dimension == self.args.vocab_size, "embedding dimension must be equal to vocab size for one_hot modes"
    
    def save_config(self):
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(self.args.__dict__, f, indent=4)
    
    def redirect_output(self):
        if not self.args.NO_REDIRECT:
            sys.stdout = open(os.path.join(self.output_dir, "output.log"), "w", buffering=1)
            sys.stderr = open(os.path.join(self.output_dir, "error.log"), "w")


def get_embedding_dir(split):
    return os.path.join(main_dir, f"embeddings/{split}_embeddings/")

def plot_loss(output_dir):
    ls = open(os.path.join(output_dir, "output.log")).readlines()
    ls = [l for l in ls if "Epoch" in l]
    ls = [l.replace(",", "").split() for l in ls]
    ls = [list(map(float, [l[1], l[3], l[6]])) for l in ls]
    ls = np.array(ls)
    plt.plot(ls[:, 1], label="Train")
    plt.plot(ls[:, 2], label="Valid")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()