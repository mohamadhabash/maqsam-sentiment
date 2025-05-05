import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.model_name = self.config["model"]["name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.config["model"]["device"],
            load_in_4bit=self.config["model"].get("load_in_4bit", False),
            trust_remote_code=True
        )

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_device(self):
        return self.device

    def get_config(self):
        return self.config
