#!/usr/bin/env python

import jax.numpy as jnp
import jax.nn as jnn
from jax import random
import numpy as np
from transformers import AutoTokenizer

from utils import RNGKeys
import json
import argparse
from tqdm import tqdm

import os
    
class DummyDataGenerator():
    """2d spiral data generator for testing"""
    def __init__(self, bsz):
        self.key = random.PRNGKey(RNGKeys().DataGenerationKey)
        self.bsz = bsz
    
    def gen_spiral(self, key, bsz):
        theta = random.uniform(key, (bsz,), minval=0.0, maxval=4*jnp.pi)

        # spiral radius (a * θ)
        a = 0.15
        r = a * theta

        # convert to x,y
        x = r * jnp.cos(theta)
        y = r * jnp.sin(theta)
        pts = jnp.stack([x, y], axis=1)

        return pts.reshape(bsz, 1, 2)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.key, key = random.split(self.key)
        x = random.normal(key, (self.bsz, 1, 2))
        y = self.gen_spiral(key, self.bsz)
        return x, y, x, y

class MnistDataGenerator():
    """Mnist data generator for testing"""
    def __init__(self, bsz, file):
        self.key = random.PRNGKey(RNGKeys().DataGenerationKey)
        self.bsz = bsz
        data = jnp.array(np.loadtxt(file, delimiter=','))
        self.labels = data[:, 0]
        self.imgs = data[:, 1:].reshape(-1, 28, 28)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        self.key, key = random.split(self.key)
        idx = random.permutation(key, jnp.arange(self.imgs.shape[0]))
        idx = idx[:self.bsz]
        img = self.imgs[idx]/255.
        y = self.labels[idx]
        noise = random.normal(key, img.shape)
        return noise, img, None, None

def helper_tokenize(tokenizer, seq_len):
    
    def tokenize_function(sentence):
        tokens = tokenizer(sentence, add_special_tokens=True)['input_ids']
        return tokens

    def trim_to_length_function(sentence):
        end_token = sentence[-1]
        return sentence[:(seq_len - 1)]+[end_token]
    
    def pad_function(sentence):
        result = [tokenizer.pad_token_id] * seq_len
        curr_len = min(len(sentence), seq_len)
        result[:curr_len] = sentence[:curr_len]
        return result

    return lambda x: pad_function(trim_to_length_function(tokenize_function(x)))

def tokenize_corpus(data_dir, seq_len, tokenizer, split):
    sentence_lst = {'src':[], 'trg': []}
    
    if split == 'train':
        path = f'{data_dir}/train.jsonl'
    elif split == 'valid':
        path = f'{data_dir}/valid.jsonl'
    elif split == 'test':
        path = f'{data_dir}/test.jsonl'
    else:
        raise ValueError(f"invalid split: {split} for dataset")

    process_func = helper_tokenize(tokenizer, seq_len)

    with open(path, 'r') as f_reader:
        for row in tqdm(f_reader):
            content = json.loads(row)
            sentence_lst['src'].append(process_func(content['src'].strip()))
            sentence_lst['trg'].append(process_func(content['trg'].strip()))
    
    sentence_lst['src'] = np.array(sentence_lst['src'])
    sentence_lst['trg'] = np.array(sentence_lst['trg'])
    out_path = os.path.abspath(path.replace(".jsonl", ".npy"))
    np.save(out_path, sentence_lst, allow_pickle=True)
    print(f'### Saved tokenized corpus with shape {sentence_lst["src"].shape} to {out_path}')

def get_embedding_matrix(data_dir, use_random_embeddings, vocab_size, embedding_dimension, random_emb_key):
    if use_random_embeddings:
        embedding_matrix = random.normal(random.PRNGKey(random_emb_key), (vocab_size, embedding_dimension))
    else:
        embedding_matrix = np.load(os.path.abspath(f'{data_dir}/embedding_matrix.npy'))
        embedding_matrix = jnp.array(embedding_matrix)
    return embedding_matrix

class QQPDataGenerator():
    def __init__(self, args, split, mode="unconditional_generation", force_single_loop=False):
        self.key = random.PRNGKey(RNGKeys().DataGenerationKey)
        self.bsz = args.bsz
        encodings = np.load(os.path.abspath(f'{args.data_dir}/{split}.npy'), allow_pickle=True).item()
        self.x_encoding = encodings['src']
        self.y_encoding = encodings['trg']
        self.mode = mode
        self.single_loop = args.single_loop or force_single_loop
        self.i = 0
        self.scale_value = args.scale_value
        self.vocab_size = args.vocab_size

        self.embedding_matrix = get_embedding_matrix(args.data_dir, args.use_random_embeddings,
            args.vocab_size, args.embedding_dimension, args.random_emb_key)
        assert self.embedding_matrix.shape[0] == self.vocab_size
        assert self.embedding_matrix.shape[1] == args.embedding_dimension
            
        # Lookup random embeddings for each sentence using token IDs
        # For example: if x_encoding[0] = [101, 2023, 2003]  (token IDs for first sentence)
        # Then x_embedding[0] = [
        #     random_embedding_matrix[101],   # random vector for token 101
        #     random_embedding_matrix[2023],  # random vector for token 2023
        #     random_embedding_matrix[2003]   # random vector for token 2003
        # ]
        self.x_embedding = self.embedding_matrix[self.x_encoding]  # (num_sentences, seq_len_x, embedding_dim)
        self.y_embedding = self.embedding_matrix[self.y_encoding]  # (num_sentences, seq_len_y, embedding_dim)
            
    def __iter__(self):
        return self
    
    def __next__(self):
        self.key, key1, key2, key3 = random.split(self.key, 4)
        if self.single_loop:
            if self.i + self.bsz >= self.x_embedding.shape[0]:
                raise StopIteration
            idx = jnp.arange(self.i, self.i + self.bsz)
            self.i += self.bsz
        else:
            idx = random.permutation(key1, jnp.arange(self.x_embedding.shape[0]))
            idx = idx[:self.bsz]
        x = self.x_embedding[idx]
        y = self.y_embedding[idx]
        x_enc = self.x_encoding[idx]
        y_enc = self.y_encoding[idx]
        noise = random.normal(key2, x.shape)
        if self.mode == "unconditional_generation":
            return noise, x, x_enc, y_enc
        elif self.mode == "seq_to_seq":
            return x, y, x_enc, y_enc
        elif self.mode == "seq_to_seq_conditional":
            return jnp.concatenate([x, noise], axis=1), jnp.concatenate([x, y], axis=1), x_enc, y_enc
        elif self.mode == "seq_to_seq_conditional_onehot":
             # One-hot encode the encodings
            x_onehot = jnn.one_hot(x_enc, num_classes=self.vocab_size)  # (bsz, seq_len_x, vocab_size)
            y_onehot = jnn.one_hot(y_enc, num_classes=self.vocab_size)  # (bsz, seq_len_y, vocab_size)
            # Reshape to match expected dimensions: flatten the one-hot dimension
            x_onehot_flat = x_onehot.reshape(x_onehot.shape[0], -1)  # (bsz, seq_len_x * vocab_size)
            y_onehot_flat = y_onehot.reshape(y_onehot.shape[0], -1)  # (bsz, seq_len_y * vocab_size)
            # Generate noise matching x_onehot shape
            noise_onehot = random.normal(key3, x_onehot_flat.shape)
            return jnp.concatenate([x_onehot_flat, noise_onehot], axis=1), jnp.concatenate([x_onehot_flat, y_onehot_flat], axis=1), x_enc, y_enc
        elif self.mode == "seq_to_seq_conditional_onehot_scaled":
            # One-hot encode the encodings
            x_onehot = jnn.one_hot(x_enc, num_classes=self.vocab_size)  # (bsz, seq_len_x, vocab_size)
            y_onehot = jnn.one_hot(y_enc, num_classes=self.vocab_size)  # (bsz, seq_len_y, vocab_size)
            # Scale: 1 -> +scale_value, 0 -> -scale_value
            x_scaled = x_onehot * (2 * self.scale_value) - self.scale_value  # Maps 1->scale_value, 0->-scale_value
            y_scaled = y_onehot * (2 * self.scale_value) - self.scale_value
            # Reshape to match expected dimensions
            x_scaled_flat = x_scaled.reshape(x_scaled.shape[0], -1)  # (bsz, seq_len_x * vocab_size)
            y_scaled_flat = y_scaled.reshape(y_scaled.shape[0], -1)  # (bsz, seq_len_y * vocab_size)
            # Generate noise matching x_scaled shape
            noise_scaled = random.normal(key3, x_scaled_flat.shape)
            return jnp.concatenate([x_scaled_flat, noise_scaled], axis=1), jnp.concatenate([x_scaled_flat, y_scaled_flat], axis=1), x_enc, y_enc
        raise ValueError(f"Unknown mode: {self.mode}")

# not used but can be used for parallelization across GPUs
def shard(batch):
    return jax.tree_map(
        lambda x: x.reshape(jax.local_device_count(), -1, *x.shape[1:]),
        batch
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="bert-base-uncased")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--split", type=str)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.config_name)
    tokenize_corpus(args.data_dir, args.seq_len, tokenizer, args.split)