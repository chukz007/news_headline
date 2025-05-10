#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH -p compute
#SBATCH -J hdln_tr
#SBATCH -t 1-23:59:59


source /home/cosuji/anaconda3/etc/profile.d/conda.sh
conda activate agent

task="translation"
#model_path="/home/support/llm/DeepSeek-R1-Distill-Llama-70B"
#model_path="unsloth/c4ai-command-a-03-2025-bnb-4bit"
model_path="CohereForAI/c4ai-command-r-plus-4bit"
model_name="llama"
dataset="/home/cosuji/spinning-storage/cosuji/NLG_Exp/clean_news_headline/PENS/personalization/pers_preprocessed.csv"
write_path="/home/cosuji/spinning-storage/cosuji/NLG_Exp/clean_news_headline/results/${model_name}"


echo "Headline Generation Started!!!"

python3 main.py --task "$task"\
			--model_path "$model_path" \
			--model_name "$model_name" \
                    	--data_path "$dataset" \
                    	--write_path "$write_path"

echo "Generation finished!!!"
