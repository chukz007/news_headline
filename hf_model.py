import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Read local .env file
hf_token = os.getenv("HF_TOKEN")

class Model(torch.nn.Module):
    def __init__(self, model_id, hf_auth=hf_token, max_length=512):
        super(Model, self).__init__()
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_auth)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.load_model(model_id, hf_auth)

    def load_model(self, model_id, hf_auth):
        #quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        quantization_config = BitsAndBytesConfig(
                                                load_in_4bit=True,
                                                bnb_4bit_quant_type="nf4",
                                                bnb_4bit_compute_dtype=torch.float16,
                                                )
        
        model = AutoModelForCausalLM.from_pretrained(
                                                    model_id,
                                                    trust_remote_code=True,
                                                    token=hf_auth,
                                                    device_map="auto",
                                                    low_cpu_mem_usage=True,
                                                    quantization_config=quantization_config,
                                                    use_safetensors=True,
                                                    )
        return model

    def generate_text(self, source):
        # Encode the input with padding and truncation
        encoded_input = self.tokenizer(
                                        source,
                                        truncation=True,
                                        padding=True,
                                        max_length=self.max_length,
                                        return_tensors="pt"
                                        )
        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                                        input_ids.to(torch.long),
                                        attention_mask=attention_mask,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        do_sample=True,
                                        max_new_tokens=512,
                                        num_beams=4,  # Increase diversity
                                        repetition_penalty=2.2,  # Adjust penalty
                                        no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                                        temperature=0.7,  # Control randomness
                                        top_k=50,  # Limit to top 50 tokens
                                        top_p=0.9,  # Nucleus sampling threshold
                                        )
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            generated_text = re.sub(r"\n+", "\n", generated_text)  # Remove excessive newline characters

        return generated_text


class Inferencer:
    def __init__(self, model):
        self.model = model

    def evaluate(self, text):
        self.model.eval()
        output = self.model.generate_text(text)
        result = output.split(text)[-1].strip().replace("\n", "  ")
        return result

    def evaluate_batch(self, texts):
        self.model.eval()
        results = []
        for text in texts:
            output = self.model.generate_text(text)
            result = output.split(text)[-1].strip().replace("\n", " ")
            results.append(result)
        return results
