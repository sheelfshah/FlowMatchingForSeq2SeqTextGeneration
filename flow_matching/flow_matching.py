#!/usr/bin/env python
import os
import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import flax.linen as nn
from flax.training import train_state
import optax
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt

from utils import main_dir, time_str, RNGKeys, Config
from models import get_model
from data_utils import get_embedding_matrix, DummyDataGenerator, MnistDataGenerator, QQPDataGenerator

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
        self.embedding_matrix = get_embedding_matrix(self.args.data_dir, self.args.use_random_embeddings,
            self.args.vocab_size, self.args.embedding_dimension, self.args.random_emb_key)
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
    
    def create_generator(self, split="train", force_single_loop=False):
        self.key, train_key = random.split(self.key)
        if self.args.DUMMY:
            return DummyDataGenerator(self.args.bsz)
        if self.args.MNIST:
            return MnistDataGenerator(self.args.bsz, os.path.join(self.main_dir, "data/mnist_train_small.csv"))
        
        return QQPDataGenerator(self.args, split, self.args.mode, force_single_loop)
    
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
    
    def create_generations(self, split, params):
        gen = self.create_generator(split, force_single_loop=True)
        sources = []
        references = []
        recoveries = []
        for x0, x1, x0enc, x1enc in gen:
            x1_pred = self.ode_solve(params, x0, x1)
            sources += [self.tokenizer.decode(x0enc[i]) for i in range(x0enc.shape[0])]
            references += [self.tokenizer.decode(x1enc[i]) for i in range(x1enc.shape[0])]
            recoveries += self.decode(x1_pred)
        return {"source": sources, "reference": references, "recover": recoveries}

    