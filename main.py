import os
import json
import argparse
from eval import evaluate_headline_performance
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


# x ="""You have these information below to help you.
# Category: {category}
# Topic: {topic}
# Entities: {entity_str}
# News Context: {context_body}

# Headline: """


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

    # task="generation"
    # model_path="llama3.2"
    # model_name="llama"
    # write_path=f"results/{model_name}"
    # data_path="archive/personalization/pers_preprocessed.csv"


    if task == "generation":
        loader = HeadlineDataLoader(data_path, prompt_template)
        prompts, news_body, ground_truth = loader.get_prompts()

        n = 5  # You can change this to len(prompts) for full processing
        output = []

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
                print(result)
                print('-' * 100)
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
                print(result)
                print('-' * 100)

        # Ensure write directory exists
        os.makedirs(write_path, exist_ok=True)

        # Save the output as a list of dictionaries in JSON format
        with open(os.path.join(write_path, "result.json"), "w", encoding="utf8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

    elif task == "translation":
        pass

    else:  # Evaluation
        pass



