#!/bin/bash

## Load conda into the shell session & Activate your environment
source /Users/chinonsoosuji/opt/anaconda3/etc/profile.d/conda.sh
conda activate lang2
dataset="archive/personalization/pers_preprocessed.csv"

# source /c/Users/CYNTHIA/anaconda3/etc/profile.d/conda.sh
# conda activate somto
# dataset="/Users/CYNTHIA/Desktop/Somto_Project/archive/personalization/pers_preprocessed.csv"

task="generation"
model_path="llama3.2"
model_name="llama"
write_path="results/${model_name}"


echo "Headline Generation Started!!!"

python main.py --task "$task"\
			--model_path "$model_path" \
			--model_name "$model_name" \
                    	--data_path "$dataset" \
                    	--write_path "$write_path"

echo "Generation finished!!!"

# To run this on your terminal, run this command first: `chmod +x run.sh` next run this `./run.sh`