from datasets import load_dataset
from data import prompt_formats

class FormatAlpaca:
    def __init__(self, split, tokenizer):
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("tatsu-lab/alpaca", split=self.split)

        self.data = self.data.filter(lambda x: x["output"].strip() != "")
        self.data = self.data.filter(lambda x: x["instruction"].strip() != "")

    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(x["instruction"].strip(), x["input"].strip()),
        "completion": prompt_formats.ANSWER.format(x["output"].strip()) + self.tokenizer.eos_token
        },
        remove_columns=["instruction", "input", "output", "text"])
        
        return data


class FormatUltraFeedback:
    def __init__(self, split, tokenizer, thresh_score=7):
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("openbmb/UltraFeedback", split=self.split)
        self.thresh_score = thresh_score
        
        self.data = self.data.filter(lambda x: len(x["completions"]) > 0)
        self.data = self.data.filter(lambda x: self.extract_best_score(x) >= self.thresh_score)

    def extract_best_score(self, sample):
        return max(r["overall_score"] for r in sample["completions"])

    def extract_best_answer(self, sample):
        return max(sample["completions"], key=lambda r: r["overall_score"])["response"]
    
    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(x["instruction"].strip(), ""),
        "completion": prompt_formats.ANSWER.format(self.extract_best_answer(x).strip()) + self.tokenizer.eos_token
        },
        remove_columns=["source", "instruction", "models", "completions", "correct_answers", "incorrect_answers"])

        return data


class FormatMMLU:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("cais/mmlu", "all", split=self.split)
    
    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(prompt_formats.build_mmlu_prompt(x["question"].strip(), x["choices"]), ""),
        "correct_response": "ABCD"[x["answer"]]
        },
        remove_columns=["subject", "choices", "answer", "question"])

        return data
    

class FormatARC:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=self.split)
    
    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(prompt_formats.build_mmlu_prompt(x["question"].strip(), x["choices"]["text"], x["choices"]["label"]), ""),
        "correct_response": x["answerKey"]
        },
        remove_columns=["id", "choices", "answerKey", "question"])

        return data


class FormatShortStories:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("PowCal/small-stories-v1", split=self.split)

        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format("Continue the following story naturally and coherently. Story so far: " + x["prompt"].strip(), "")
        },
        remove_columns=["response"])
        
        return data
    
    @property
    def base_format(self):

        data = self.data.map(lambda x: {
        "prompt": x["prompt"].strip()
        },
        remove_columns=["response"])
        
        return data
    

class FormatSmallPrompts():
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("PowCal/small-prompts-v1", split=self.split)

        self.data = self.data.filter(lambda x: x["instruction"].strip() != "")

    @property
    def sft_format(self):
        data = self.data.map(
            lambda x: {
                "prompt": prompt_formats.PROMPT.format(x["instruction"].strip(), ""),
            }
        )
        return data
    
    @property
    def base_format(self):

        data = self.data.map(lambda x: {
            "prompt": x["instruction"].strip() + " Okay, here goes: "
            }
        )
        
        return data


class FormatNoveltyBench:
    def __init__(self, split):
        self.split = split
        
        self.data = load_dataset("yimingzhang/novelty-bench", split=self.split)

        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(x["prompt"].strip(), "")
        },
        remove_columns=["id"])
        
        return data
    
    @property
    def base_format(self):

        data = self.data.map(lambda x: {
        "prompt": x["prompt"].strip()
        },
        remove_columns=["id"])
        
        return data


class FormatMATH500:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = load_dataset("HuggingFaceH4/MATH-500", split="test")

        self.data = self.data.filter(lambda x: x["problem"].strip() != "")
        self.data = self.data.filter(lambda x: x["answer"].strip() != "")

    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT_COT.format(x["problem"].strip()),
        "completion": prompt_formats.ANSWER_COT.format(x["answer"].strip()) + self.tokenizer.eos_token,
		"correct_solution": prompt_formats.to_boxed(x['answer'])
        },
        remove_columns=["problem", "solution", "answer", "subject", "level", "unique_id"])
        
        return data


class FormatGSM8K:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("openai/gsm8k", "main", split=self.split)

        self.data.filter(lambda x: self.extract_hash_reason(x) is not None and self.extract_hash_answer(x) is not None)

    def extract_xml_answer(self, text):
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    def extract_hash_reason(self, text):
        if "####" not in text:
            return None
        return text.split("####")[0].strip()

    def extract_hash_answer(self, text):
        if "####" not in text:
            return None
        return text.split("####")[1].strip()

    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT_COT.format(x["question"].strip()),
        "completion": prompt_formats.ANSWER_COT.format(self.extract_hash_answer(x["answer"])) + "",
        "correct_solution": self.extract_hash_answer(x["answer"]).strip()
		},
        remove_columns=["question", "answer"])

        return data


class FormatMinerva:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = load_dataset("math-ai/minervamath", split="test")

        self.data = self.data.filter(lambda x: x["question"].strip() != "")
        self.data = self.data.filter(lambda x: x["answer"].strip() != "")

    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT_COT.format(x["question"].strip()),
        "completion": prompt_formats.ANSWER_COT.format(x["answer"].strip()) + self.tokenizer.eos_token,
		"correct_solution": prompt_formats.to_boxed(x['answer'])
        },
        remove_columns=["question", "answer"])
        
        return data


class FormatNumina:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data = load_dataset("PowCal/NuminaMath-CoT-100k", split="train")

        self.data = self.data.filter(lambda x: x["problem"].strip() != "")
        self.data = self.data.filter(lambda x: x["solution"].strip() != "")

    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT_COT.format(x["problem"].strip()),
        "completion": prompt_formats.ANSWER_COT.format(x["solution"].strip()) + self.tokenizer.eos_token
        },
        remove_columns=["problem", "solution", "source"])
        
        return data


class FormatEval:
    def __init__(self, data):
        self.data = data

    def clean_instruction(self, prompt):
        if "<|instruct|>" in prompt and "<|/instruct|>" in prompt:
            instruction = prompt.split("<|instruct|>")[1].split("<|/instruct|>")[0].strip()
            return instruction
        return prompt

    def clean_response(self, response):
        if "<|response|>" in response:
            response = response.split("<|response|>")[1]
        if "<|/response|>" in response:
            response = response.split("<|/response|>")[0]
        return response.strip()

    @property
    def judge_instruct_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.JUDGE_INSTRUCT.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
    
    @property
    def bleu_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            formatted_data.append({'instruction': cleaned_instruction, 'responses': cleaned_responses})
        return formatted_data

    @property
    def judge_story_format(self):
        formatted_data = []
        for entry in self.data:
            cleaned_instruction = self.clean_instruction(entry['prompt'])
            cleaned_responses = [self.clean_response(response) for response in entry['responses']]
            judge_prompts = [prompt_formats.JUDGE_STORY.format(cleaned_instruction, cleaned_response) for cleaned_response in cleaned_responses]
            formatted_data.append({'instruction': cleaned_instruction, 'response': cleaned_responses, 'judge_prompt': judge_prompts})
        return formatted_data
    

class FormatTruthfulQA:
    def __init__(self, split, tokenizer):
        self.split = split
        self.tokenizer = tokenizer
        self.data = load_dataset("domenicrosati/TruthfulQA", split=self.split)

        self.data = self.data.filter(lambda x: x["Best Answer"].strip() != "")
        self.data = self.data.filter(lambda x: x["Question"].strip() != "")


    @property
    def sft_format(self):

        data = self.data.map(lambda x: {
        "prompt": prompt_formats.PROMPT.format(x["Question"].strip(), ""),
        "completion": prompt_formats.ANSWER.format(x["Best Answer"].strip()) + self.tokenizer.eos_token
        },
        remove_columns=["Type", "Category", "Correct Answers", "Incorrect Answers", "Source", "Question", "Best Answer"])

        return data

def to_bbox(ans):
    ans = ans.strip()
    return f"\\boxed{{{ans}}}"


class FormatMaliciousInstruct:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("PowCal/MaliciousInstruct", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):
        data = self.data.map(lambda x: {
            "prompt": prompt_formats.PROMPT.format(x["prompt"].strip(), ""),
        })
        return data


class FormatHarmBench:
    def __init__(self, split):
        self.split = split
        self.data = load_dataset("PowCal/HarmBench100", split=self.split)
        self.data = self.data.filter(lambda x: x["prompt"].strip() != "")

    @property
    def sft_format(self):
        data = self.data.map(lambda x: {
            "prompt": prompt_formats.PROMPT.format(x["prompt"].strip(), ""),
        }, remove_columns=["__index_level_0__"])
        return data
