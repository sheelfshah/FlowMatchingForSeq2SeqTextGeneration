#!/usr/bin/env python
import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from jax import random
import jax

from utils import RNGKeys

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -jnp.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
    )
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding

def get_1d_sincos_pos_embed(embed_dim, length, temperature=10000.0):
    """
    Create 1D sinusoidal position embeddings (similar to Transformer).
    
    :param embed_dim: dimension of the embedding
    :param length: length of the sequence
    :param temperature: temperature for the sinusoidal functions
    :return: (length, embed_dim) position embeddings
    """
    position = jnp.arange(length, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(jnp.arange(0, embed_dim, 2, dtype=jnp.float32) * 
                       -(jnp.log(temperature) / embed_dim))
    
    pos_embed = jnp.zeros((length, embed_dim))
    pos_embed = pos_embed.at[:, 0::2].set(jnp.sin(position * div_term))
    pos_embed = pos_embed.at[:, 1::2].set(jnp.cos(position * div_term))
    
    return pos_embed

class FlowMLP(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x, t):
        # Flatten x and concatenate with t
        t = jnp.expand_dims(t, axis=-1)  # (batch, 1)
        x_flat = jnp.reshape(x, (x.shape[0], -1))  # (batch, dim)
        x_with_t = jnp.concatenate([x_flat, t], axis=-1) # (batch, dim + 1)

        # MLP layers
        h = nn.Dense(self.hidden_dim)(x_with_t)
        h = nn.silu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.silu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.silu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.silu(h)
        h = nn.Dense(self.dim)(h)

        # Reshape to output
        out = jnp.reshape(h, (x.shape[0], 1, self.dim))
        return out


class AdaLN(nn.Module):
    """Adaptive Layer Normalization - conditions on timestep."""
    hidden_dim: int
    
    @nn.compact
    def __call__(self, x, t_emb):
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            t_emb: (batch, time_emb_dim)
        Returns:
            Modulated x with shift and scale from timestep
        """
        # Layer norm, normalizes across hidden_dim
        x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        
        # Get shift and scale from time embedding
        modulation = nn.Dense(2 * self.hidden_dim)(t_emb)  # (batch, 2*hidden_dim)
        shift, scale = jnp.split(modulation, 2, axis=-1)  # Each (batch, hidden_dim)
        
        # Apply: scale * norm(x) + shift
        shift = shift[:, None, :]  # (batch, 1, hidden_dim)
        scale = scale[:, None, :]  # (batch, 1, hidden_dim)
        
        return (1.0 + scale) * x + shift

class DiTBlock(nn.Module):
    """DiT transformer block with adaptive normalization."""
    hidden_dim: int
    num_heads: int
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x, t_emb, train: bool = True):
        """
        h = attention(norm(x))
        y = x + mlp(norm(h))
        * norm is actually AdaLN, and there is dropout at multiple places
        Args:
            x: (batch, seq_len, hidden_dim)
            t_emb: (batch, time_emb_dim)
        """
        # Self-attention with AdaLN
        h = AdaLN(self.hidden_dim)(x, t_emb)
        h = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=not train,
            kernel_init=nn.initializers.xavier_uniform(),
        )(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
        x = x + h  # Residual
        
        # MLP with AdaLN
        h = AdaLN(self.hidden_dim)(x, t_emb)
        mlp_dim = int(self.hidden_dim * self.mlp_ratio)
        h = nn.Dense(mlp_dim)(h)
        h = nn.gelu(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.Dropout(rate=self.dropout_rate)(h, deterministic=not train)
        x = x + h  # Residual
        
        return x


class DiTFlowModel(nn.Module):
    """
    DiT-style model for flow matching.
    Simpler than U-Net but still effective with proper time conditioning.
    """
    input_dim: int = 128
    output_dim: int = 128
    len_dim: int = 64
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    time_emb_dim: int = 256
    
    @nn.compact
    def __call__(self, x, t, train: bool = True):
        """
        Args:
            x: (batch, seq_len, input_dim)
            t: (batch,)
        Returns:
            (batch, seq_len, output_dim) - Predicted velocity
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.time_emb_dim)
        t_emb = nn.Dense(self.time_emb_dim)(t_emb)
        t_emb = nn.silu(t_emb)
        t_emb = nn.Dense(self.time_emb_dim)(t_emb)
        
        # Input projection
        h = nn.Dense(self.hidden_dim)(x)

        # Position embeddings
        pos_emb = get_1d_sincos_pos_embed(self.hidden_dim, self.len_dim)
        h = h + pos_emb[None, :, :]
        
        # Transformer blocks
        for _ in range(self.num_layers):
            h = DiTBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout_rate=self.dropout_rate
            )(h, t_emb, train)
        
        # Final layer norm and projection
        h = AdaLN(self.hidden_dim)(h, t_emb)
        h = nn.Dense(self.output_dim, kernel_init=nn.initializers.zeros)(h)
        
        return h

def get_model(args):
    if args.model == "DiTFlowModel":
        model = DiTFlowModel(
            input_dim=args.embedding_dimension,
            output_dim=args.embedding_dimension,
            len_dim=args.len_dim*(2 if args.mode == "seq_to_seq_conditional" else 1),
            hidden_dim=args.model__hidden_dim,
            num_layers=args.model__num_layers,
            num_heads=args.model__num_heads,
            mlp_ratio=args.model__mlp_ratio,
            dropout_rate=args.model__dropout_rate,
            time_emb_dim=args.model__time_emb_dim
        )
    elif args.model == "FlowMLP":
        model = FlowMLP(
            args.embedding_dimension,
            args.model__hidden_dim
        )
    else:
        raise ValueError("Invalid model type")
    dummy_x = jnp.zeros((1, args.len_dim*(2 if args.mode == "seq_to_seq_conditional" else 1), args.embedding_dimension))
    dummy_t = jnp.zeros((1,))
    variables = model.init(random.PRNGKey(RNGKeys().ModelInitKey), dummy_x, dummy_t)
    if jax.local_device_count() > 1:
        variables = jax.device_put_replicated(variables, jax.local_devices())
    return model, variables

def get_embedding_matrix():
    return jnp.array(np.load("diffusion_models/embedding_matrix.npy"))