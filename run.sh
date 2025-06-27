#!/bin/bash

## Load conda into the shell session & Activate your environment
##source /Users/chinonsoosuji/opt/anaconda3/etc/profile.d/conda.sh
##source /c/Users/CYNTHIA/anaconda3/etc/profile.d/conda.sh
source /home/chinonso/anaconda3/etc/profile.d/conda.sh
conda activate somto

task="translati"
model_path="command-r-plus"  ##"llama3.2"
model_name="dpsk"
dataset="archive/personalization/pers_preprocessed.csv"
write_path="results/${model_name}"

echo "Started!!!"

python main.py --task "$task" \
    --model_path "$model_path" \
    --model_name "$model_name" \
    --data_path "$dataset" \
    --write_path "$write_path"

echo "Finished!!!"

# To run this: chmod +x run.sh && ./run.sh