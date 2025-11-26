import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

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

HEIGHT_ASPECT_RATIO = 0.618 # Golden ratio for aesthetically pleasing aspect
sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

def process_logs(logs_dir, name=None):
    with open(f"{logs_dir}/output.log", 'r') as f:
        lines = f.readlines()
    
    training_lines = [line for line in lines if line.startswith('Step')]
    validation_lines = [line for line in lines if (line.startswith('Validation') or line.startswith('Step'))]
    training_df = pd.DataFrame(columns=['Step', 'Loss', 'LR'])
    validation_df = pd.DataFrame(columns=['Step', 'EMA_Factor', 'BLEU', 'ROUGE-L', 'Dist1', 'AvgLen'])
    for line in training_lines:
        # extract numbers from the str "Step 0, Loss: 0.886265, LR: 0.000000"
        step = int(line.split('Step ')[1].split(',')[0])
        loss = float(line.split('Loss: ')[1].split(',')[0])
        lr = float(line.split('LR: ')[1].split(',')[0])
        training_df = training_df._append({'Step': step, 'Loss': loss, 'LR': lr}, ignore_index=True)
    
    latest_step = 0
    for line in validation_lines:
        if line.startswith('Step'):
            latest_step = int(line.split('Step ')[1].split(',')[0])
            continue
        # extract numbers from the str "Validation: BLEU: 0.000395, ROUGE-L: 0.002335, Dist1: 0.980156, AvgLen: 55.706055"
        ema_factor = float(line.split('EMA ')[1].split(':')[0]) if "EMA" in line else 0
        bleu = float(line.split('BLEU: ')[1].split(',')[0])
        rougel = float(line.split('ROUGE-L: ')[1].split(',')[0])
        dist1 = float(line.split('Dist1: ')[1].split(',')[0])
        avg_len = float(line.split('AvgLen: ')[1].split(',')[0])
        validation_df = validation_df._append({'Step': latest_step, 'EMA_Factor': ema_factor, 'BLEU': bleu, 'ROUGE-L': rougel, 'Dist1': dist1, 'AvgLen': avg_len}, ignore_index=True)
    
    training_df['name'] = logs_dir.split('/')[-1] if name is None else name
    validation_df['name'] = logs_dir.split('/')[-1] if name is None else name
    return training_df, validation_df


folders = {
    "Pretrained embeddings": "diffusion_models/dit_4m_s2sC_pte_20251125_143456",
    "Pretrained embeddings, Time Encoding Period 100": "diffusion_models/dit_4m_s2sC_pte_tep100_20251125_144829",
    "Pretrained embeddings, with EMA": "diffusion_models/dit_4m_s2sC_pte_ema_20251125_195437",
    "Random embeddings": "diffusion_models/dit_4m_s2sC_re_20251125_143658",
    "Random embeddings, embedding dim = 64": "diffusion_models/dit_4m_s2sC_re_64_20251125_144028",
    "Random embeddings, embedding dim = 256": "diffusion_models/dit_4m_s2sC_re_256_20251125_144200"
}

train_dfs = {}
val_dfs = {}
for name, folder in folders.items():
    train_dfs[name], val_dfs[name] = process_logs(folder, name)

# Compare Pretrained embeddings and Pretrained embeddings, Time Encoding Period 100
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=val_dfs['Pretrained embeddings'], x='Step', y='BLEU', label='Time Encoding Period 10000')
sns.lineplot(data=val_dfs['Pretrained embeddings, Time Encoding Period 100'], x='Step', y='BLEU', label='Time Encoding Period 100')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
plt.legend()
plt.title('Comparing BLEU scores with different time encoding periods')
plt.savefig("plots/tep_comparision.pdf", format='pdf', bbox_inches='tight')
plt.show()

# Compare pretrained embeddings vs random embeddings
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=val_dfs['Pretrained embeddings'], x='Step', y='BLEU', label='Pretrained embeddings')
sns.lineplot(data=val_dfs['Random embeddings'], x='Step', y='BLEU', label='Random embeddings')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
plt.legend()
plt.title('Comparing BLEU scores with and without pretrained embeddings')
plt.savefig("plots/emb_comparision.pdf", format='pdf', bbox_inches='tight')
plt.show()

# Compare embedding dimensions
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=val_dfs['Random embeddings, embedding dim = 64'], x='Step', y='BLEU', label='Embedding dimension = 64')
sns.lineplot(data=val_dfs['Random embeddings'], x='Step', y='BLEU', label='Embedding dimension = 128')
sns.lineplot(data=val_dfs['Random embeddings, embedding dim = 256'], x='Step', y='BLEU', label='Embedding dimension = 256')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
plt.legend()
plt.title('Comparing BLEU scores with different embedding dimensions')
plt.savefig("plots/emb_dim_comparision.pdf", format='pdf', bbox_inches='tight')
plt.show()

# Compare EMA factors
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=val_dfs['Pretrained embeddings, with EMA'].loc[val_dfs['Pretrained embeddings, with EMA']['EMA_Factor'] == 0], x='Step', y='BLEU', label='Without EMA')
sns.lineplot(data=val_dfs['Pretrained embeddings, with EMA'].loc[val_dfs['Pretrained embeddings, with EMA']['EMA_Factor'] == 0.999], x='Step', y='BLEU', label='EMA factor = 0.999')
sns.lineplot(data=val_dfs['Pretrained embeddings, with EMA'].loc[val_dfs['Pretrained embeddings, with EMA']['EMA_Factor'] == 0.9999], x='Step', y='BLEU', label='EMA factor = 0.9999')
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x/1000)}k'))
plt.legend()
plt.title('Comparing BLEU scores with different EMA factors')
plt.savefig("plots/ema_comparision.pdf", format='pdf', bbox_inches='tight')
plt.show()