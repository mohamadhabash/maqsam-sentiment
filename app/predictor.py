import re
import torch
from app.model_loader import ModelLoader

class SentimentPredictor:
    def __init__(self):
        self.loader = ModelLoader()
        self.model = self.loader.get_model()
        self.tokenizer = self.loader.get_tokenizer()
        self.device = self.loader.get_device()
        self.config = self.loader.get_config()

        self.prompt_template = self.config["prompt"]["base"]
        self.task_prompt = self.config["task"]["sentiment"]

    def build_prompt(self, text: str) -> str:
        question = f"{self.task_prompt}\n\nText: \"{text}\""
        return self.prompt_template.format_map({'Question': question})

    def infer(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        input_len = input_ids.shape[-1]
        generate_ids = self.model.generate(
            input_ids,
            top_p=self.config["model"]["top_p"],
            temperature=self.config["model"]["temperature"],
            max_new_tokens=self.config["model"]["max_new_tokens"],
            min_length=input_len + self.config["model"]["min_length_buffer"],
            repetition_penalty=self.config["model"]["repetition_penalty"],
            do_sample=True,
        )
        response = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return response.split("### Response :")[-1].strip()

    def extract_sentiment(self, response: str) -> str:
        match = re.search(r"(Positive|Neutral|Negative)", response, re.IGNORECASE)
        return match.group(0).capitalize() if match else "Neutral"

    def predict(self, text: str) -> str:
        prompt = self.build_prompt(text)
        response = self.infer(prompt)
        return self.extract_sentiment(response)
