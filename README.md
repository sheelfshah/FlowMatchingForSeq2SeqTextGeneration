# Flow Matching for Sequence-to-Sequence Text Generation

This project implements flow matching for sequence-to-sequence text generation tasks.
We use continuous representations of text sequences to enable direct transfer of diffusion techniques from image generation to text generation.

## Environment Setup

Run the commands in the setup script:
```bash
git clone https://github.com/sheelfshah/FlowMatchingForSeq2SeqTextGeneration.git
cd FlowMatchingForSeq2SeqTextGeneration
cat scripts/setup.sh
# you could also just source this script but it is safer to see the commands before running them
```

## Data Setup
The data was downloaded from the DiffuSeq repository link: https://drive.google.com/drive/folders/1BHGCeHRZU7MQF3rsqXBIOCU2WIC3W6fb

The `.npy` files in `datasets/QQP/` are tokenized versions of the data. To generate them, run:
```bash
# assuming data is in datasets/QQP/
bash scripts/tokenize_data.sh
```

## Acknowledgement

This repo is based on the DiffuSeq project (https://github.com/Shark-NLP/DiffuSeq). However, most code was revamped to run in JAX/Flax environment, and only the data processing pipeline has been retained. The pretrained embedding from DiffuSeq is extracted as a .npy file and used for experiments.