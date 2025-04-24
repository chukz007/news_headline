import os
import json
import argparse
from eval import evaluate_headline_performance
from model import Model, Inferencer
from load_dataset import HeadlineDataLoader
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())  # Read local .env file
hf_token = os.getenv("HF_TOKEN")


prompt_template = """Role: News Headline Generator
Task: Generate a compelling news headline using the given news article.
Please generate only one short, concise, and accurate news headline that captures the essence of the news and engages the audience. Do not generate any other thing asides the task given.

News Body: 
{newsbody}

Headline: """

x """You have these information below to help you.
Category: {category}
Topic: {topic}
Entities: {entity_str}
News Context: {context_body}

Headline: """


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

    #task = "generation"
    #model_path = "/home/support/llm/Llama-3.3-70B-Instruct"
    #model_id = "/home/support/llm/Llama-3.1-70B-Instruct"
    #data_path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/news_headline/PENS/personalization/pers_preprocessed.csv"
    #write_path = "/home/cosuji/spinning-storage/cosuji/NLG_Exp/news_headline/results"

    if task == "generation":
        #Load the dataset
        loader = HeadlineDataLoader(data_path, prompt_template)

        #Add the prompt to the dataset
        prompts, news_body, ground_truth = loader.get_prompts()

        #Load the Hugging-face Model
        model = Model(model_id=model_id, hf_auth=hf_token, max_length=256)
    
        # Create the inferencer
        inferencer = Inferencer(model)

        pred = []
        n = 5
        for prompt in prompts[:n]:
            #print(prompt)
            result = inferencer.evaluate(prompt).split(prompt)[-1]
            pred.append(result)
            print(result)
            print('-'*100)


        #create a json file with the news_body, prediction and ground_truth as column names
        output = {"news_body": news_body[:n],
                  "prediction": pred,
                  "ground_truth": ground_truth[:n]}

        # Ensure the write directory exists
        os.makedirs(write_path, exist_ok=True)

        #Write your result to a json file - result.json
        with open(os.path.join(write_path, "result.json"), "w", encoding="utf8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


        for filename, contents in output.items():
            with open(os.path.join(write_path, f"{filename}.txt"), "w", encoding="utf8") as f:
                f.write('\n\n'.join(str(item) for item in contents))  # separate each item with line breaks
    
    elif task == "translation":
        pass

    else: #Evaluation
        pass



