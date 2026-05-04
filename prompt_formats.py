

def to_boxed(ans):
    ans = ans.strip()
    return f"\\boxed{{{ans}}}"


PROMPT = """
Below is an instruction that describes a task. Write a response that appropriately completes the request. Input might contain additional information.
Instruction: 
<|instruct|> 
{} 
<|/instruct|> 

Input: 
<|input|> 
{} 
<|/input|> 

Response: 
<|response|>
"""

ANSWER = """ {} 
<|/response|>
"""


PROMPT_REASON = """
Below is an instruction that describes a task. Write a response that appropriately completes the request. Respond in the following format: 
<|reasoning|> 
... 
<|/reasoning|> 
<|answer|> 
... 
<|/answer|> 

Instruction: 
<|instruct|> 
{} 
<|/instruct|> 

Response: 
<|response|> 
"""


ANSWER_REASON = """<|reasoning|> 
{} 
<|/reasoning|> 
<|answer|> 
{} 
<|/answer|> 
<|/response|> 
"""


JUDGE_INSTRUCT = """You will be given a user_instruction and a model_response pair.

Your task is to provide a 'total rating' evaluating how well the model_response fulfills the user_instruction.

Be strict in your assessment. Give a high score only if the response:
- Fully addresses the instruction
- Is coherent and well-structured
- Maintains a style and tone appropriate to the instruction

Deduct points if the response:
- Fails to follow the instruction
- Introduces errors or inconsistencies
- Includes irrelevant or nonsensical content
- Contradicts the context or previous information
- Is empty or contains no meaningful content

Pay attention to the number of sentences or symbols: if the response has more or fewer sentences or symbols than explicitly requested in the user_instruction, apply **-2** to the total score.

However, **do not penalize** the response for being incomplete or ending abruptly **due to model token limits** — only penalize if the content itself is flawed.

Give your answer as an INTEGER on a scale of 0 to 5, where:
- 0 = completely unhelpful, irrelevant, or empty response
- 5 = fully correct, coherent, and appropriately styled response

Use the following response format exactly:

Total rating: (your rating, as an INT between 0 and 5)

Now here are the user_question and system_answer.
user_question: {}
system_answer: {}
"""

JUDGE_STORY = """You will be given a story_beginning and story_continuation pair.
Your task is to provide a 'total rating' evaluating how well the story_continuation continues the story_beginning.

Be strict in your assessment. Give a high score only if the continuation is coherent, stylistically consistent, and narratively natural.
If the continuation changes the point of view, tone, or style, or breaks logical consistency with the beginning — deduct points accordingly.
If it ends abruptly, contradicts earlier events, or feels disconnected, apply further deductions.
However, **do not penalize** the continuation if it ends mid-sentence or abruptly **due to a model token limit** — only penalize if the ending itself is logically or stylistically flawed.

Give your answer as an INTEGER on a scale of 0 to 5, where:
- 0 = incoherent, off-topic, or nonsensical continuation
- 5 = seamless, natural, and stylistically consistent continuation

Use the following response format exactly:

Total rating: (your rating, as an INT between 0 and 5)

Now here are the story_beginning and story_continuation.
story_beginning: {}
story_continuation: {}
"""


MMLU_INSTRUCT = """
Answer the following multiple choice question. Your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of choice labels.

{}

{}
"""


PROMPT_COT = """
Solve the following math problem step by step. Show only the necessary reasoning and give the final answer within \\boxed{{}}.
Problem: 
<|problem|> 
{} 
<|/problem|> 

Solution: 
<|solution|>
"""

ANSWER_COT = """ {} 
<|/solution|>
"""


def build_mmlu_prompt(question: str, choices: list, labels=None):
    if labels is None:
        labels = [chr(ord('A') + i) for i in range(len(choices))]

    choice_block = "\n".join(f"{label.strip()}) {choice.strip()}" for label, choice in zip(labels, choices))

    prompt = MMLU_INSTRUCT.format(question, choice_block)
    return prompt.strip()
