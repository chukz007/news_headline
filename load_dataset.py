import pandas as pd
import ast
import random

class HeadlineDataLoader:
    def __init__(self, path, prompt_template):
        self.path = path
        self.prompt_template = prompt_template
        self.df = self.load_data()

    def load_data(self):
        try:
            df = pd.read_csv(self.path, sep='\t', on_bad_lines='skip')
            df['context'] = df['context'].apply(ast.literal_eval)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def prepare_prompt(self, row):
        try:
            entity_dict = ast.literal_eval(row["Title entity"])
            entity_str = ', '.join([f"{key} -> {value}" for key, value in entity_dict.items()])

            if isinstance(row['context'], list) and len(row['context']) >= 5:
                selected_context = random.sample(row['context'], 5)
            else:
                selected_context = row['context'][:5]

            context_body = ', '.join(selected_context)
            prompt = self.prompt_template.format(
                category=row['Category'],
                topic=row['Topic'],
                entity_str=entity_str,
                context_body=context_body,
                newsbody=row['News body']
            )
            return prompt
        except Exception as e:
            print(f"Error processing row: {e}")
            return None

    def get_prompts(self):
        prompts, news_body, ground_truth  = [], [], []
        for index, row in self.df.iterrows():
            prompt = self.prepare_prompt(row)
            news_body.append(row["News body"])
            ground_truth.append(row["Headline"])
            if prompt:
                prompts.append(prompt)
        return prompts, news_body, ground_truth
