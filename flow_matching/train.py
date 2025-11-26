#!/usr/bin/env python
from jax import random, jit, value_and_grad

from tqdm import tqdm

from utils import main_dir, time_str, RNGKeys
from models import save_checkpoint, update_ema_params
from flow_matching import FlowMatching, mse_loss
from eval import eval

FM = FlowMatching(main_dir, time_str)
key = random.PRNGKey(RNGKeys().MainLoopKey)

def create_train_step(model):

    def train_step(state, x_t, v_t, t):
        def loss_fn(params):
            u_t = model.apply({'params': params}, x_t, t)
            return ((u_t - v_t)**2).mean()
        
        loss, grads = value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        state = update_ema_params(state)
        return state, loss
    
    return jit(train_step)


if FM.args.split == 'train':
    
    # Create train state
    state = FM.create_train_state()
    train_step = create_train_step(FM.model)
    generator = FM.create_generator(FM.args.split)
    for step  in range(FM.args.num_steps):
        x_0, x_1, _, _ = next(generator)
        key, train_key = random.split(key)

        t = random.uniform(train_key, (x_0.shape[0],))
        t_unsqz = t[:, None, None]
        x_t = (1-t_unsqz)*x_0 + t_unsqz*x_1
        v_t = x_1 - x_0
        state, loss = train_step(state, x_t, v_t, t)
        
        if step % FM.args.print_interval == 0:
            tqdm.write(f'Step {step}, Loss: {loss:.6f}, LR: {FM.create_schedule()(state.step):.6f}')
        
        if step % FM.args.checkpointing_interval == 0:
            val_gens = FM.create_generations("valid", state.params)
            bleu, rougel, dist1, avg_len = eval(val_gens)
            tqdm.write(f'Validation: BLEU: {bleu:.6f}, ROUGE-L: {rougel:.6f}, Dist1: {dist1:.6f}, AvgLen: {avg_len:.6f}')
            for ema_fac, ema_param in state.ema_params.items():
                val_gens = FM.create_generations("valid", ema_param)
                bleu, rougel, dist1, avg_len = eval(val_gens)
                tqdm.write(f'Validation EMA {ema_fac}: BLEU: {bleu:.6f}, ROUGE-L: {rougel:.6f}, Dist1: {dist1:.6f}, AvgLen: {avg_len:.6f}')
            
            save_checkpoint(step, state.params, FM.cfg.output_dir, state.ema_params)
    
    save_checkpoint(step, state.params, FM.cfg.output_dir, state.ema_params)

