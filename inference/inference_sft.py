import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import json
from huggingface_hub import upload_file

from data import formatting


class Inference:
    def __init__(self, base_model, dataset_name, dataset_format, peft_weights, output_dir, repo_id, max_new_tokens, temperature, top_p, do_sample, num_return_sequences):

        self.base_model = base_model
        self.dataset_name = dataset_name
        self.dataset_format = dataset_format
        self.peft_weights = peft_weights
        self.output_dir = output_dir
        self.repo_id = repo_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences

        self.id = f"{self.peft_weights.split('/')[-1]}-{self.dataset_name}-{self.dataset_format}" if self.peft_weights else f"{self.base_model.split('/')[-1]}-{self.dataset_name}-{self.dataset_format}"
        self.output_dir = os.path.join(output_dir, self.id)
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Loading base model: {self.base_model}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        self.model = AutoModelForCausalLM.from_pretrained(self.base_model)

        if self.dataset_name == "nb":
            self.dataset = formatting.FormatNoveltyBench(split='curated')
        elif self.dataset_name == "ss":
            self.dataset = formatting.FormatShortStories(split='train')
        elif self.dataset_name == "sp":
            self.dataset = formatting.FormatShortPoems(split='train')
        elif self.dataset_name == "smpr":
            self.dataset = formatting.FormatSmallPrompts(split='eval')
        elif self.dataset_name == "mmlu":
            self.dataset = formatting.FormatMMLU(split='test')
        elif self.dataset_name == "arc":
            self.dataset = formatting.FormatARC(split='test')
        elif self.dataset_name == "gsm":
            self.dataset = formatting.FormatGSM8K(split='test')
        elif self.dataset_name == "mi":
            self.dataset = formatting.FormatMaliciousInstruct(split='train')
        elif self.dataset_name == "hb":
            self.dataset = formatting.FormatHarmBench(split='train')
        elif self.dataset_name == "math500":
            self.dataset = formatting.FormatMATH500(tokenizer=self.tokenizer)
        elif self.dataset_name == "minerva":
            self.dataset = formatting.FormatMinerva(tokenizer=self.tokenizer)
        else:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")
        
        if self.dataset_format == "sft":
            self.dataset = self.dataset.sft_format
        elif self.dataset_format == "base":
            self.dataset = self.dataset.base_format
        else:
            raise ValueError(f"Invalid dataset format: {self.dataset_format}")

        if self.peft_weights:
            print(f"Applying PEFT adapter from: {self.peft_weights}")
            self.model = self.load_peft_model(self.peft_weights)

        self.model.eval()

        

    def load_peft_model(self, path):
        """Load a PEFT adapter from local dir or Hugging Face repo (with optional subfolder)."""
        if os.path.isdir(path):
            return PeftModel.from_pretrained(self.model, path)

        parts = path.strip("/").split("/")
        if len(parts) == 1:
            repo_id, subfolder = parts[0], None
        elif len(parts) == 2:
            repo_id, subfolder = "/".join(parts), None
        else:
            repo_id, subfolder = "/".join(parts[:2]), "/".join(parts[2:])

        return PeftModel.from_pretrained(self.model, repo_id, subfolder=subfolder)
    


    def run(self):

        output_file = os.path.join(self.output_dir, "responses.jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            pass

        for item in tqdm(self.dataset):

            inputs = self.tokenizer(item["prompt"], return_tensors="pt").to(self.model.device)
            inputs.pop("token_type_ids", None)
            prompt_length = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    num_return_sequences=self.num_return_sequences,
                )

            generated_tokens = outputs[:, prompt_length:]
            responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            # store all responses
            entry = {
                "prompt": item["prompt"],
                "responses": responses
            }

            if "correct_response" in item.keys():
                entry["correct_response"] = item["correct_response"]
            if "correct_solution" in item.keys():
                entry["correct_solution"] = formatting.to_bbox(item["correct_solution"])
         
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.repo_id is not None:
            print(f"Pushing resulting file to {self.repo_id}")
            upload_file(path_or_fileobj=output_file, path_in_repo=f"{self.id}/responses.jsonl", repo_id=self.repo_id, repo_type="dataset")


def main():
    parser = argparse.ArgumentParser(description="Stream inference from a base model with optional PEFT weights")
    parser.add_argument("--base_model", type=str, required=True, help="Path or name of the base model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset")
    parser.add_argument("--dataset_format", type=str, required=True, help="Dataset format")
    parser.add_argument("--peft_weights", type=str, default=None, help="Path to PEFT weights")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--repo_id", type=str, default=None, help="HF repository to push to")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="max new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="top p")
    parser.add_argument("--do_sample", choices=["True", "False"], default="True", help="do sample (greedy decoding if False)")
    parser.add_argument("--num_return_sequences", type=int, default=10, help="num sequences to generate")
    args = parser.parse_args()

    inference = Inference(
        base_model=args.base_model, 
        dataset_name=args.dataset,
        dataset_format=args.dataset_format,
        peft_weights=args.peft_weights,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample == "True",
        num_return_sequences=args.num_return_sequences
    )

    inference.run()



if __name__ == "__main__":
    main()
