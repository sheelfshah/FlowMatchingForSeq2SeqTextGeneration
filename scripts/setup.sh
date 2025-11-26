#!/bin/bash

conda create -n FlowMatchingForSeq2SeqTextGeneration python=3.10
conda activate FlowMatchingForSeq2SeqTextGeneration

pip install --upgrade "jax[cuda12]" flax optax
pip install transformers
pip install tqdm
pip install torchmetrics nltk
pip install matplotlib seaborn
pip install pandas

conda env export > environment.yml