#!/usr/bin/env python
import os
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import flax.linen as nn
from flax.training import train_state, checkpoints, common_utils
import optax
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt

from utils import main_dir, time_str, RNGKeys, Config, get_embedding_dir
from models import get_model, get_embedding_matrix
from data_utils import load_embeddings, DummyDataGenerator, MnistDataGenerator, QQPDataGenerator

import itertools
import os
from tqdm import tqdm
from functools import partial
import json

def nearest_token_rounding(model_emb, text_emb):
    """
    Args:
        model_emb: (vocab_size, embedding_dim)
        text_emb: (seqlen, embedding_dim)
    Returns:
        ((seqlen, embedding_dim), (seqlen,)) 
        rounded tokens and their indices
    """
    # use ||x-y||_2^2 = ||x||_2^2 + ||y||_2^2 - 2<x,y>
    # then do argmin over x to get the nearest token
    # since the argmin over x doesn't depend on ||y||_2^2, we can ignore it
    norm_model_emb = jnp.linalg.norm(model_emb, axis=-1, keepdims=True) # (vocab_size, 1)
    # norm_text_emb = jnp.linalg.norm(text_emb, axis=-1, keepdims=True) # (seqlen, 1)
    dist = norm_model_emb**2 - 2 * jnp.dot(model_emb, text_emb.T) # + norm_text_emb.T**2 # (vocab_size, seqlen)
    nn_idx = jnp.argmin(dist, axis=0) # (seqlen,)
    rounded_tokens = model_emb[nn_idx] # (seqlen, embedding_dim)
    return rounded_tokens, nn_idx

def batch_nearest_token_rounding(model_emb, text_emb):
    """
    Args:
        model_emb: (vocab_size, embedding_dim)
        text_emb: (bsz, seqlen, embedding_dim)
    Returns:
        ((bsz, seqlen, embedding_dim), (bsz, seqlen,)) 
        rounded tokens and their indices
    """
    bsz, seqlen, _ = text_emb.shape
    rounded_tokens, nn_idx = nearest_token_rounding(model_emb, text_emb.reshape(-1, text_emb.shape[-1]))
    rounded_tokens = rounded_tokens.reshape(bsz, seqlen, text_emb.shape[-1])
    nn_idx = nn_idx.reshape(bsz, seqlen)
    return rounded_tokens, nn_idx

def save_checkpoint(step, params, cfg, args, force=False):
    valid_step = force or (step + 1) % args.checkpointing_interval == 0
    if args.write_model and valid_step and params is not None:
        print(f"Saving checkpoint at step {step + 1}")
        checkpoints.save_checkpoint(
            ckpt_dir=cfg.output_dir,
            target=params,
            step=step,
            prefix='model_flow_',
            overwrite=force,
            keep=1000 # keep last 100 checkpoints
        )

def mse_loss_individual(pred, target):
    return jnp.mean((pred - target) ** 2, axis=tuple(range(1, pred.ndim)))

def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

class FlowMatching:
    def __init__(self, main_dir, time_str):
        self.main_dir = main_dir
        self.time_str = time_str
        self.cfg = Config(main_dir, time_str)
        self.cfg.redirect_output()
        self.cfg.save_config()
        self.args = self.cfg.args
        
        devices = jax.devices()
        print(f"Available JAX devices: {devices}")
        print(f"Device count: {len(devices)}")
        
        np.random.seed(RNGKeys().FMInitKey)
        self.key = random.PRNGKey(RNGKeys().FMInitKey)

        self.load_model()
        self.load_checkpoint()
        self.load_embedding_matrix()
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.config_name)
        self.load_optimizer()
    
    def load_model(self):
        self.model, self.variables = get_model(self.args)
        total_params = sum(x.size for x in jax.tree_util.tree_leaves(self.variables['params']))
        print(f"Model created with total parameters: {total_params}")
    
    def update_params(self, params):
        self.variables['params'] = params
    
    def load_checkpoint(self):
        if self.args.checkpoint_dir != "":
            print("Loading model from checkpoint")
            checkpoint_dir = os.path.join(self.main_dir, f"diffusion_models/{self.args.checkpoint_dir}")
            restored_params = checkpoints.restore_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=None, #variables['params'],
                prefix='model_flow_'
            )
            self.update_params(restored_params)
    
    def load_embedding_matrix(self):
        self.embedding_matrix = get_embedding_matrix()
        print("Embedding matrix loaded")
        
    def create_schedule(self):
        return optax.schedules.warmup_exponential_decay_schedule(
            init_value=1e-7,
            peak_value=self.args.lr,
            warmup_steps=self.args.warmup_steps,
            transition_steps=self.args.transition_steps,
            decay_rate=self.args.decay_rate,
            transition_begin=self.args.transition_begin,
            staircase=False
        )
    
    def load_optimizer(self):
        if self.args.DUMMY:
            self.tx = optax.adam(self.args.lr)
        else:
            self.tx = optax.chain(
                optax.clip_by_global_norm(self.args.max_grad_norm),
                optax.adam(self.create_schedule())
            )
    
    def create_generator(self, split="train"):
        self.key, train_key = random.split(self.key)
        if self.args.DUMMY:
            return DummyDataGenerator(self.args.bsz)
        if self.args.MNIST:
            return MnistDataGenerator(self.args.bsz, os.path.join(self.main_dir, "data/mnist_train_small.csv"))
        
        return QQPDataGenerator(self.args, get_embedding_dir(split), self.args.mode)
    
    def create_train_state(self):
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.variables['params'],
            tx=self.tx
        )
        if jax.local_device_count() > 1:
            state = jax.device_put_replicated(state, jax.local_devices())
        return state
    
    @partial(jax.jit, static_argnums=(0,))
    def forward_pass(self, params, x_t, t):
        """Single forward pass."""
        # kwargs = {'train': False} if not self.args.DUMMY else {}
        u_t = self.model.apply({'params': params}, x_t, t, train=False)
        return u_t

    # jit keeping self and num_steps constant
    # @partial(jax.jit, static_argnums=(0, 4))
    def ode_solve(self, params, x_0, x_1, num_steps=100):
        """Solves the ODE using the forward pass."""
        t = jnp.zeros(x_0.shape[0])
        x_t = x_0
        dt = 1 / num_steps
        for i in range(num_steps):
            u_t = self.forward_pass(params, x_t, t)
            x_t = x_t + u_t * dt
            t = t + dt
            if self.args.mode == "seq_to_seq_conditional":
                x_t = x_t.at[:, :self.args.len_dim].set(x_0[:, :self.args.len_dim])
        return x_t
    
    def decode(self, x, full_decode=False):
        if (not full_decode) and self.args.mode == "seq_to_seq_conditional":
            x = x[:, self.args.len_dim:]
        _, nn_idx = batch_nearest_token_rounding(self.embedding_matrix, x)
        return [self.tokenizer.decode(nn_idx[i]) for i in range(nn_idx.shape[0])]

if __name__ == "__main__":
    flow_matching = FlowMatching(main_dir, time_str)
    gen = flow_matching.create_generator(flow_matching.args.split)
    sources = []
    references = []
    recoveries = []
    for x0, x1, x0enc, x1enc in tqdm(gen):
        x1_pred, _ = flow_matching.ode_solve(flow_matching.variables['params'], x0, x1)
        sources += [flow_matching.tokenizer.decode(x0enc[i]) for i in range(x0enc.shape[0])]
        references += [flow_matching.tokenizer.decode(x1enc[i]) for i in range(x1enc.shape[0])]
        recoveries += flow_matching.decode(x1_pred, full_decode=False)
    
    with open(f"{flow_matching.cfg.output_dir}/{flow_matching.args.split}_generations.json", "w") as f:
        for source, reference, recovery in zip(sources, references, recoveries):
            json.dumps({"source": source, "reference": reference, "recover": recovery}, file=f)

    # x0 = flow_matching.train_x_embedding[:10]         
    # x1 = flow_matching.train_y_embedding[:10]
    # print([flow_matching.tokenizer.decode(row) for row in flow_matching.train_x_encoding[:10]])         
    # print([flow_matching.tokenizer.decode(row) for row in flow_matching.train_y_encoding[:10]])         
    # # x1_hat = flow_matching.ode_solve(flow_matching.variables['params'], x0, x1, 1000)
    # print("=="*30)
    # # print(flow_matching.decode(x1_hat))
    