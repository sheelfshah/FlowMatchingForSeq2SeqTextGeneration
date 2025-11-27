import os, sys, glob, json
import numpy as np
import argparse

# import torch
from torchmetrics.text.rouge import ROUGEScore

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

from utils import main_dir, time_str
from flow_matching import FlowMatching, mse_loss
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "text.usetex": False,      # Tell Matplotlib to use LaTeX for all text
    "font.family": "serif",   # Use serif fonts (like Computer Modern)
    "font.serif": ["Computer Modern Roman"], # Specify the font
    "font.size": 10,          # Match the main paper font size (often 10pt)
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
})

sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)


def get_bleu(recover, reference):
    return sentence_bleu([reference.split()], recover.split(), smoothing_function=SmoothingFunction().method4,)

def distinct_n_gram(hypn,n):
    dist_list = []
    for hyp in hypn:
        hyp_ngrams = []
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
        total_ngrams = len(hyp_ngrams)
        unique_ngrams = len(list(set(hyp_ngrams)))
        if total_ngrams == 0:
            return 0
        dist_list.append(unique_ngrams/total_ngrams)
    return  np.mean(dist_list)

    hyp_ngrams = []
    for hyp in hypn:
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(list(set(hyp_ngrams)))
    if total_ngrams == 0:
        return 0
    dist_n = unique_ngrams/total_ngrams
    return  dist_n

def eval(generation_dict, sos='[CLS]', eos='[SEP]', sep='[SEP]', pad='[PAD]', direct_input=False, direct_output=False):
    """
    Args:
        generation_dict: {"source": List[str], "reference": List[str], "recover": List[str]}
    Returns:
        bleu: float
        rougel: float
        dist1: float
        avg_len: float
    """
    def clean_text(text):
        return text.replace(eos, '').replace(sos, '').replace(sep, '').replace(pad, '').strip()

    sources = list(map(clean_text, generation_dict['source']))
    references = list(map(clean_text, generation_dict['reference']))
    recovers = list(map(clean_text, generation_dict['recover']))

    bleu = []
    rougel = []
    dist1 = []
    avg_len = []

    for source, reference, recover in zip(sources, references, recovers):
        if direct_input:
            recover = source
        elif direct_output:
            recover = reference

        avg_len.append(len(recover.split(' ')))
        bleu.append(get_bleu(recover, reference))
        rougel.append(ROUGEScore()(recover, reference)['rougeL_fmeasure'].tolist())
        dist1.append(distinct_n_gram([recover], 1))

    return np.mean(bleu), np.mean(rougel), np.mean(dist1), np.mean(avg_len)

if __name__ == '__main__':
    FM = FlowMatching(main_dir, time_str)
    generator = FM.create_generator("valid", True)
    x_0, x_1, _, _ = next(generator)
    t = jnp.zeros((x_0.shape[0],))
    num_points = 100
    dt = 1/num_points
    losses = jnp.zeros(num_points+1)
    times = jnp.zeros(num_points+1)
    v_t = x_1 - x_0
    for i in range(num_points+1):
        t_unsqz = t[:, None, None]
        x_t = (1-t_unsqz)*x_0 + t_unsqz*x_1
        u_t = FM.forward_pass(FM.variables['ema_params'][0.9999], x_t, t)
        losses = losses.at[i].set(mse_loss(u_t, v_t))
        times = times.at[i].set(t[0])
        t += dt
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=np.array(times), y=np.array(losses))
    ax.set_xlabel('t')
    ax.set_ylabel('Loss')
    plt.title('Loss vs t')
    plt.savefig("plots/loss_vs_t.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # sys.exit()

    results = {
        1: {},
        2: {},
        4: {},
        8: {},
        16: {},
        32: {},
        64: {},
        100: {},
        200: {},
        500: {},
        1000: {},
        2000: {}
    }
    for n in results.keys():
        val_gens = FM.create_generations("valid", FM.variables['ema_params'][0.9999], num_steps = n)
        bleu, rougel, dist1, avg_len = eval(val_gens)
        results[n]["valid"] = {"gens": val_gens}
        results[n]["valid"]["bleu"] = bleu
        results[n]["valid"]["rougel"] = rougel
        results[n]["valid"]["dist1"] = dist1
        results[n]["valid"]["avg_len"] = avg_len
        print(f"Num steps: {n}, BLEU: {bleu:.6f}, ROUGE-L: {rougel:.6f}, Dist1: {dist1:.6f}, AvgLen: {avg_len:.6f}")
        test_gens = FM.create_generations("test", FM.variables['ema_params'][0.9999], num_steps = n)
        bleu, rougel, dist1, avg_len = eval(test_gens)
        results[n]["test"] = {"gens": test_gens}
        results[n]["test"]["bleu"] = bleu
        results[n]["test"]["rougel"] = rougel
        results[n]["test"]["dist1"] = dist1
        results[n]["test"]["avg_len"] = avg_len
        print(f"Num steps: {n}, BLEU: {bleu:.6f}, ROUGE-L: {rougel:.6f}, Dist1: {dist1:.6f}, AvgLen: {avg_len:.6f}")
    
    os.makedirs(os.path.join(FM.main_dir, f"plots/eval_{FM.args.checkpoint_dir}"), exist_ok=True)
    json.dump(results, open(os.path.join(FM.main_dir, f"plots/eval_{FM.args.checkpoint_dir}/eval_results.json"), 'w'))
    