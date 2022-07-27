from utils import Config
from preprocess import preprocess_file

from pathlib import Path
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Union

GPU_ID = "1"
device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")


class PromptModel():
    def __init__(self, config: Config):
        self.config = config
        self.data = preprocess_file(
                    file_path = Path(self.config.data_dir, self.config.data_file),
                    task = self.config.task_name,
                    dataset = self.config.dataset_name
                )
        self.init_model(self.config.model)


    def init_model(self, model_name: str):
        if model_name == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)




if __name__ == "__main__":
    config = Config()
    model = PromptModel(config) 
