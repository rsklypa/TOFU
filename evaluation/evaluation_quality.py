import argparse
import os
import json
import re
import torch
from huggingface_hub import hf_hub_download, upload_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


from data import formatting


class Evaluator:
    def __init__(self, judge_model, input_file, output_dir, repo_id, max_new_tokens):

        self.judge_model = judge_model
        self.input_file = input_file
        self.repo_id = repo_id
        self.max_new_tokens = max_new_tokens

        self.model_id = f"{self.input_file.split('/')[-2]}"
        self.output_dir = os.path.join(output_dir, self.model_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Loading judge model: {self.judge_model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.judge_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.judge_model)
        self.model.eval()

        print(f"Loading data from: {self.input_file}")
        data = self.load_file(self.input_file)
        dataset_format = self.input_file.split('/')[-2].split('-')[-1]
        
        if dataset_format == 'sft':
            self.data = formatting.FormatEval(data).judge_instruct_format
        elif dataset_format == 'base':
            self.data = formatting.FormatEval(data).judge_story_format
        else:
            raise ValueError(f"Invalid dataset format: {dataset_format}")
        print(f"Loaded {len(self.data)} instruction-response pairs to score")


    def load_file(self, path):
            
        if not os.path.isdir(path):
            print(f"Downloading from {path} ...")
            parts = path.strip("/").split("/")
            repo_id, subpath = "/".join(parts[:2]), "/".join(parts[2:])
            path = hf_hub_download(
                repo_id=repo_id,
                filename=subpath,
                repo_type="dataset"
            )
            print(f"Downloaded to: {path}")

        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

        return data
    

    def extract_rating(self, text):

        match = re.search(r"Total rating:\s*([0-9]+(?:\.\d+)?)", text)

        if not match:
            print(f"Warning: Failed to extract rating, generated text snippet: {text}")
            return None

        rating = float(match.group(1))
        return rating if 0 <= rating <= 5 else None
        
    
    
    def score_single(self, judge_prompt):
        
        chat = self.tokenizer.apply_chat_template([{"role": "user", "content": judge_prompt}], tokenize=False)

        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        inputs.pop("token_type_ids", None)
        prompt_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
      
        generated_tokens = outputs[:, prompt_length:]
        judge_response = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        rating = self.extract_rating(judge_response)
        
        return rating, judge_response
    

    def run(self):

        output_file = os.path.join(self.output_dir, "responses_quality_scores.jsonl")
        with open(output_file, "w", encoding="utf-8") as f: pass

        for idx, item in enumerate(tqdm(self.data, desc="Scoring responses")):

            score_results = {'rating': [], 'judge_response': []}
            ratings, judge_responses = zip(*(self.score_single(p) for p in item['judge_prompt']))
            score_results['rating'], score_results['judge_response'] = list(ratings), list(judge_responses)
            
            valid_ratings = [r for r in score_results['rating'] if r is not None]
            score_results['average_rating'] = sum(valid_ratings) / len(valid_ratings) if valid_ratings else None
            
            result = item | score_results
            subset_keys = ['response', 'rating', 'judge_response']
            result['scored_responses'] = [dict(zip(subset_keys, v)) for v in zip(*(result[k] for k in subset_keys))]
            result.pop('judge_prompt')
            result.pop('response')
            result.pop('rating')
            result.pop('judge_response')

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"\nScoring complete! Results saved to: {output_file}")

        if self.repo_id is not None:
            print(f"Pushing resulting file to {self.repo_id}")
            upload_file(path_or_fileobj=output_file, path_in_repo=f"{self.model_id}/responses_quality_scores.jsonl", repo_id=self.repo_id, repo_type="dataset")

def main():
    parser = argparse.ArgumentParser(description="Score responses using a judge model with formatting.py")
    parser.add_argument('--input_file', type=str, required=True, help='Path to local JSONL file with prompts and responses')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save scored responses')
    parser.add_argument('--judge_model', type=str, required=True, help='Judge model to use for scoring')
    parser.add_argument("--repo_id", type=str, default=None, help="HF repository to push to")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="max new tokens")
    args = parser.parse_args()
    
    evaluator = Evaluator(
        judge_model=args.judge_model,
        input_file=args.input_file,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        max_new_tokens=args.max_new_tokens
    )
    
    evaluator.run()


if __name__ == "__main__":
    main()
