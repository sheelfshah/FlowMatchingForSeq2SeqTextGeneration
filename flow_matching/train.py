#!/usr/bin/env python
from jax import random, jit, value_and_grad

from tqdm import tqdm

from utils import main_dir, time_str, RNGKeys
from flow_matching import save_checkpoint, FlowMatching, mse_loss

FM = FlowMatching(main_dir, time_str)
key = random.PRNGKey(RNGKeys().MainLoopKey)

def create_train_step(model):

    def train_step(state, x_t, v_t, t):
        def loss_fn(params):
            u_t = model.apply({'params': params}, x_t, t)
            return ((u_t - v_t)**2).mean()
        
        loss, grads = value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
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
        
        save_checkpoint(step, state.params, FM.cfg, FM.args)
    
    save_checkpoint(step, state.params, FM.cfg, FM.args, force=True)

