import argparse
import os
import json

from tqdm import tqdm
from huggingface_hub import hf_hub_download, upload_file
from sacrebleu.metrics import BLEU

from data import formatting


class Evaluator:
    def __init__(self, input_file, output_dir, repo_id):

        self.input_file = input_file
        self.repo_id = repo_id
        self.model_id = f"{self.input_file.split('/')[-2]}"

        # --- 1. Creating output directory ---
        self.output_dir = os.path.join(output_dir, self.model_id)
        os.makedirs(self.output_dir, exist_ok=True)

        # --- 2. Loading data to score ---
        print(f"Loading data from: {self.input_file}")
        data = self.load_file(self.input_file)
        self.data = formatting.FormatEval(data).bleu_format

        # --- 3. Scorer parameters ---
        self.bleu = BLEU(tokenize='13a', lowercase=True, effective_order=True)


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


    def self_bleu(self, responses):
        scores = []

        for i, hyp in enumerate(responses):
            refs = [responses[j] for j in range(len(responses)) if j != i]
            score = self.bleu.sentence_score(hyp, refs)
            scores.append(score.score)

        return sum(scores) / len(scores)
    

    def run(self):

        output_file = os.path.join(self.output_dir, "responses_diversity_scores.jsonl")
        with open(output_file, "w", encoding="utf-8") as f: pass
      
        for idx, item in enumerate(tqdm(self.data, desc="Scoring responses")):

            cleaned_responses = [response.replace("\n", " ") for response in item['responses']]
        
            score_results = {'bleu': self.self_bleu(cleaned_responses)}

            result = item | score_results
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"\nScoring complete! Results saved to: {output_file}")

        if self.repo_id is not None:
            print(f"Pushing resulting file to {self.repo_id}")
            upload_file(path_or_fileobj=output_file, path_in_repo=f"{self.model_id}/responses_diversity_scores.jsonl", repo_id=self.repo_id, repo_type="dataset")


def main():
    parser = argparse.ArgumentParser(description="Score responses diversity with self BLEU")
    parser.add_argument('--input_file', type=str, required=True, help='Path to JSONL file with prompts and responses')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save scored responses')
    parser.add_argument("--repo_id", type=str, default=None, help="HF repository to push to")
    
    args = parser.parse_args()
    
    evaluator = Evaluator(
        input_file=args.input_file,
        output_dir=args.output_dir,
        repo_id=args.repo_id
    )
    
    evaluator.run()


if __name__ == "__main__":
    main()
