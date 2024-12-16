#!/bin/bash

# Activate conda environment if needed
# conda activate your_environment_name
# Install dependencies
pip install -r requirements.txt --user

# Optional: Uncomment and configure as needed
# wandb login --verify your_wandb_token

bash script/4x-Llama3-GPT2-Large.sh
bash script/5b_4x_large_llama.sh
