from utils import Config
from preprocess import preprocess_file
from prompts import generate_prompts

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
        """ Initialize tokenizers and models Initialize tokenizers and models  
        """
        if model_name == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-large").to(device)


    def generate(self):
        """ Method to prepare prompts and generate.
        """
        prompts, gold = generate_prompts(self.data, self.config)    # Generate prompts and their answers
        for ix, prompt in enumerate(prompts):
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            with torch.no_grad():
                outputs = self.model.generate(input_ids.to(device), num_return_sequences=5, num_beams=5, output_scores=True, return_dict_in_generate=True)
            #print(outputs)
            print(outputs.sequences_scores)
            exit()
            print(f"Prompt :{prompt}")
            print(f"Gold: {gold[ix]}")
            print(f"Generation: {self.tokenizer.decode(outputs[0], skip_special_tokens=True)}")
            print()
            if ix == 50:
                exit()


if __name__ == "__main__":
    config = Config()
    model = PromptModel(config)
    model.generate()
