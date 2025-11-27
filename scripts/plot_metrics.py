import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import json

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


diffuseq_steps = [1,2,5,10,25,50,125,250,500,1000,2000]
diffuseq_bleu_values = [6.56245400962304e-05,5.945714060152421e-05,8.77340802966579e-05,6.494396251445722e-05,8.538547610352802e-05,0.00012633103207594976,0.0003651276775399245,0.0018444334707758798,0.010838288636346049,0.142284006724347,0.1862149695304189]
diffuseq_rouge_values = [0.00045321190729737283,0.00043125718981027605,0.0005967552088201046,0.0004482587866485119,0.0006785382807254791,0.0008139704629778862,0.002350968936830759,0.013623140358179808,0.09127995198518038,0.4643362303122878,0.531826974517107]

flowmatching_json = json.load(open("plots/eval_dit_big_s2sC_pte_ema_tep100_20251126_012914/copied_ckpts/eval_results.json"))
flowmatching_steps = [int(step) for step in flowmatching_json.keys()]
flowmatching_bleu_values = [flowmatching_json[str(step)]['test']['bleu'] for step in flowmatching_steps]
flowmatching_rouge_values = [flowmatching_json[str(step)]['test']['rougel'] for step in flowmatching_steps]

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(x=diffuseq_steps, y=diffuseq_bleu_values, label='DiffuSeq', marker='o')
sns.lineplot(x=flowmatching_steps, y=flowmatching_bleu_values, label='FlowMatching', marker='^')
ax.set_xscale('log')
ax.set_xlabel('NFE')
ax.set_ylabel('BLEU')
plt.legend()
plt.title('BLEU Scores vs NFE')
plt.savefig("plots/bleu_comparision.pdf", format='pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(x=diffuseq_steps, y=diffuseq_rouge_values, label='DiffuSeq', marker='o')
sns.lineplot(x=flowmatching_steps, y=flowmatching_rouge_values, label='FlowMatching', marker='^')
ax.set_xscale('log')
ax.set_xlabel('NFE')
ax.set_ylabel('ROUGE-L')
plt.legend()
plt.title('ROUGE-L Scores vs NFE')
plt.savefig("plots/rouge_comparision.pdf", format='pdf', bbox_inches='tight')
plt.show()