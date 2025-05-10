import os
import json
import argparse
from ollama_model import OllamaModel
from load_dataset import HeadlineDataLoader
from dotenv import load_dotenv, find_dotenv


# _ = load_dotenv(find_dotenv())  # Read local .env file
# hf_token = os.getenv("HF_TOKEN")

system_prompt = """You are a professional journalist specializing in headline writing. You have a strong understanding of news structure and style, and can produce headlines that are clear, accurate, and compelling.

Your task is to generate a short headline (under 12 words) that:
1. Clearly reflects the article’s core message or main event
2. Preserves important context, including key people, actions, and consequences
3. Matches the tone and style of professional journalism across various categories (e.g., politics, business, sports, lifestyle)
4. May include brief quotes if directly relevant
5. Avoids generic phrasing, vagueness, or repetition
 
Output:
Only the final headline. Do not include any explanation or additional text.
"""

prompt_template = """You are given a news article. Your task is to generate a concise, informative, and compelling headline that accurately summarizes the core event or message. The headline should be under 12 words and capture key names, actions, or outcomes.

The article may belong to any category (e.g., politics, business, sports, lifestyle, crime, health). Use the article's content to infer the appropriate tone and focus.

Instructions:
1. Read the article carefully.
2. Identify the central event, key people involved, and the outcome.
3. Write a short headline that reflects the article's main point with clarity and relevance.
4. Return only the headline — no commentary or extra text.

News Body:
{newsbody}

Headline:
"""


def save_json(data, path, filename):
    """
    Save a list of dictionaries as a JSON file.

    Args:
        data (list): The list of dictionaries to save.
        path (str): The directory where the file should be saved.
        filename (str): The name of the JSON file.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    with open(full_path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="Specify the task you wish to do - generation/translation/evaluation")
    parser.add_argument("--model_path", help="path to the model in hugging face or local path")
    parser.add_argument("--model_name", help="Name of the model")
    parser.add_argument("--data_path", help="path to the dataset")
    parser.add_argument("--write_path", help="path to write the results")
    args = parser.parse_args()

    task = args.task
    model_id = args.model_path
    model_name = args.model_name
    data_path = args.data_path
    write_path = args.write_path
    
    # Ensure write directory exists
    os.makedirs(write_path, exist_ok=True)

    # task="generation"
    # model_path="llama3.2"
    # model_name="llama"
    # write_path=f"results/{model_name}"
    # data_path="archive/personalization/pers_preprocessed.csv"


    if task == "generation":
        loader = HeadlineDataLoader(data_path, prompt_template)
        prompts, news_body, ground_truth = loader.get_prompts()

        n = 5  # You can change this to len(prompts) for full processing
        output, comet_output = [], []

        if "llm" in model_id:
            from hf_model import Model, Inferencer
            model = Model(model_id=model_id, hf_auth=hf_token, max_length=256)
            inferencer = Inferencer(model)

            for i in range(n):
                result = inferencer.evaluate(prompts[i]).split(prompts[i])[-1].strip()
                output.append({
                    "news_body": news_body[i],
                    "prediction": result,
                    "ground_truth": ground_truth[i]
                })

        else:
            ollama_model = OllamaModel(model_name=model_id, temperature=0)
            model = ollama_model.model_(system_prompt)

            for i in range(n):
                result = model.invoke({'input': prompts[i]}).content.strip()
                output.append({
                    "news_body": news_body[i],
                    "prediction": result,
                    "ground_truth": ground_truth[i]
                })
                # print(result)
                # print('-' * 100)
        
        save_json(output, write_path, "result.json")

    elif task == "translation":
        pass

    else:  # Evaluation
        from eval import evaluate_headline_performance, evaluate_with_comet, evaluate_semantic_metrics

        # # Run headline-level metrics
        # print("\nEvaluating headline quality (BLEU, METEOR, ROUGE)...")
        # avg_bleu, avg_meteor, avg_rouge_f1 = evaluate_headline_performance(
        #     json_path=os.path.join(write_path, "result.json")
        # )

        # print("Running COMET evaluation...")
        # avg_comet, comet_scores = evaluate_with_comet(
        #     json_path=os.path.join(write_path, "result.json"),
        #     gpus=1
        # )

        # metric_result = {
        #     "average_bleu": avg_bleu,
        #     "average_meteor": avg_meteor,
        #     "average_rouge_f1": avg_rouge_f1,
        #     # "average_comet": avg_comet,
        #     "comet_scores": comet_scores
        # }

        # save_json(metric_result, write_path, "metric_result.json")
        # print(f"\nAll metrics saved to {os.path.join(write_path, 'metric_result.json')}")
        
        # Now compute BLEURT + BERTScore only
        print("Evaluating semantic similarity (BERTScore, BLEURT)...")
        semantic_metrics = evaluate_semantic_metrics(json_path=os.path.join(write_path, "result.json"))

        # Save separately
        save_json(semantic_metrics, write_path, "semantic_metrics.json")
        print(f"\nBLEURT + BERTScore metrics saved to {os.path.join(write_path, 'semantic_metrics.json')}")



