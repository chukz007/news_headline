import os
import json
import argparse
from dotenv import load_dotenv, find_dotenv
from utilities.ollama_model import OllamaModel
from utilities.hf_model import Model, Inferencer
from utilities.load_dataset import HeadlineDataLoader
from utilities.prompts import prompt_template, system_prompt
from utilities.translate import translate_with_hf, translate_with_ollama
from utilities.eval import evaluate_headline_performance, evaluate_with_comet, evaluate_semantic_metrics, evaluate_with_comet_referenceless

_ = load_dotenv(find_dotenv())  # Read local .env file
hf_token = os.getenv("HF_TOKEN")

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

        if "/" in model_id:
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
        input_path = os.path.join(write_path, "result.json")
        sample_size = 1000  # You can change this to len(prompts) for full processing
        if "/" in model_id:
            translated = translate_with_hf(input_path, model_id, hf_token, sample_size)
        else:
            translated = translate_with_ollama(input_path, model_id, sample_size)

        print("Saving translated results...")
        save_json(translated, write_path, f"{task}_result.json")
    
    elif task == "translation evaluation":

        translated_path = os.path.join(write_path, "translation_result.json")
        output_path = os.path.join(write_path, "translation_comet_scores.json")

        with open(translated_path, "r", encoding="utf8") as f:
            data = json.load(f)

        target_languages = [
            "French", "Spanish", "Portuguese", "Italian",
            "German", "Russian", "Chinese", "Hindi", "Arabic"
        ]

        results = {}

        for lang in target_languages:
            lang_code = lang[:2].lower()
            prediction_key = f"prediction_{lang_code}"

            if not all(prediction_key in item for item in data):
                print(f"Skipping {lang} ({lang_code}): missing translations.")
                continue

            comet_input = [
                {"src": item["prediction_en"], "mt": item[prediction_key]}
                for item in data if prediction_key in item
            ]

            print(f"\nEvaluating {lang} with COMET (referenceless)...")
            comet_scores, avg_score = evaluate_with_comet_referenceless(
                data=comet_input,
                model_name="Unbabel/wmt22-cometkiwi-da",
                batch_size=8,
                gpus=1
            )

            results[f"comet_{lang_code}"] = round(avg_score, 3)

        save_json(results, write_path, "translation_comet_scores.json")
        print(f"\nSaved COMET scores to {output_path}")

    else:  # Evaluation
        # # Run headline-level metrics
        print("\nEvaluating headline quality (BLEU, METEOR, ROUGE)...")
        avg_bleu, avg_meteor, avg_rouge_f1 = evaluate_headline_performance(
            json_path=os.path.join(write_path, "result.json")
        )

        # print("Running COMET evaluation...")
        comet_scores, avg_comet = evaluate_with_comet(json_path=os.path.join(write_path, "result.json"), gpus=1)

        # Now compute BLEURT + BERTScore only
        print("Evaluating semantic similarity (BERTScore, BLEURT)...")
        avg_bertscore, avg_bleurt = evaluate_semantic_metrics(json_path=os.path.join(write_path, "result.json"))


        metric_result = {
            "average_bleu": round(avg_bleu, 3),
            "average_meteor": round(avg_meteor, 3),
            "average_rouge_f1": round(avg_rouge_f1, 3),
            "comet_scores": round(avg_comet, 3),
            "average_bertscore_f1": round(avg_bertscore, 3),
            "average_bleurt": round(avg_bleurt, 3),
        }


        save_json(metric_result, write_path, "metric_result.json")
        print(f"\nAll metrics saved to {os.path.join(write_path, 'metric_result.json')}")
        

        # Save separately
        #save_json(semantic_metrics, write_path, "semantic_metrics.json")
        #print(f"\nBLEURT + BERTScore metrics saved to {os.path.join(write_path, 'semantic_metrics.json')}")
        # comet-score -s src.txt -t hyp1.txt --model Unbabel/wmt22-cometkiwi-da

