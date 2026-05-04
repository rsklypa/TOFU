import argparse
import os
import json
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download, upload_file
from math_utils import *


class Evaluator:
    def __init__(self, dataset_name, input_file, output_dir=None, repo_id=None):
        self.dataset_name = dataset_name
        self.repo_id = repo_id
        self.model_id = input_file.split("/")[-2]
        self.output_dir = os.path.join(output_dir, self.model_id) if output_dir else None
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        print(f"Loading data from: {input_file}")
        self.data = self.load_predictions(input_file)

    def load_predictions(self, path):
        if not os.path.isfile(path):
            path = hf_hub_download(repo_id=self.repo_id, filename=path, repo_type="dataset")
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def run(self):
        benches = {
            "math500": self.math500,
            "minerva": self.minerva,
            "gsm8k":   self.gsm8k,
        }
        benches[self.dataset_name]()

    def score(self, desc, gold_fn, pred_fn, eq_fn):
        accuracies = []
        max_scores = []

        for item in tqdm(self.data, desc=desc):
            responses = item.get("responses", [])
            gold_raw = item.get("correct_solution")
            if gold_raw is None:
                continue
            gold_list = gold_fn(gold_raw)
            if not gold_list:
                continue
            correct = 0
            any_correct = 0
            for resp in responses:
                pred_list = pred_fn(resp)
                for pred in pred_list:
                    if pred is None:
                        continue
                    if any(eq_fn(pred, g) for g in gold_list):
                        correct += 1
                        any_correct = 1
                        break  

            acc_i = correct / len(responses) if responses else 0
            accuracies.append(acc_i)
            max_scores.append(any_correct)
            
        mean = float(np.mean(accuracies)) if accuracies else 0.0
        max_acc = float(np.mean(max_scores)) if max_scores else 0.0

        return mean, max_acc

    def push(self, mean, max_acc):
        print(f"\n Mean accuracy: {mean:.4f}")
        print(f" Max accuracy (pass@16): {max_acc:.4f}")
        if self.output_dir:
            score_path = os.path.join(self.output_dir, "score.jsonl")
            with open(score_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"Pass@k": max_acc, "Mean": mean}) + "\n")
            if self.repo_id is not None:
                upload_file(
                    path_or_fileobj=score_path,
                    path_in_repo=f"{self.model_id}/score.jsonl",
                    repo_id=self.repo_id,
                    repo_type="dataset",
                )

    def math500(self):
        mean, max_acc = self.score(desc="Scoring Math500", gold_fn=lambda g: [normalize_latex(extract_boxed(g))], pred_fn=lambda r: [normalize_latex(extract_boxed(r))], eq_fn=math_equal_symbolic)
        self.push(mean, max_acc)

    def minerva(self):
        mean, max_acc = self.score(desc="Scoring MinervaMath", gold_fn=lambda g: [normalize(extract_boxed(g))], pred_fn=lambda r: [normalize(extract_boxed(r))], eq_fn=math_equal)
        self.push(mean, max_acc)

    def gsm8k(self):
        mean, max_acc = self.score(desc="Scoring GSM8K", gold_fn=lambda g: [normalize(extract_boxed(g))], pred_fn=lambda r: [normalize(extract_boxed(r))], eq_fn=math_equal)
        self.push(mean, max_acc)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math500", "minerva"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--split",  type=str, default="test")
    args = parser.parse_args()

    evaluator = Evaluator(
        dataset_name=args.dataset,
        input_file=args.input_file,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        split=args.split,
    )
    evaluator.run()

if __name__ == "__main__":
    main()
  
