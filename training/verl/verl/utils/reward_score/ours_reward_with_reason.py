import random
import re
import string
import openai
from openai import OpenAI
import os

USER_PROMPT_with_reason = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. 

- First, please refer to the provided explanation to assist you to answer the question. 

- Then, please assess the claim's consistency with the document by responding with either "Yes" or "No". Please wrap your final answer in <answer> and </answer>.

Document: [DOCUMENT]
Claim: [CLAIM]
Explanation: [EXPLANATION]
"""

USER_PROMPT_ori = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. 

- Please assess the claim's consistency with the document by responding with either "Yes" or "No". Please wrap your final answer in <answer> and </answer>.

Document: [DOCUMENT]
Claim: [CLAIM]
"""

def think_format_reward(predict_str: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str.strip())
    return 1.0 if match_result else 0.0

    
def format_reward(predict_str: str) -> float:
    pattern = re.compile(r"^<think>[^<>]*</think>\s*<reason>[^<>]*</reason>\s*<answer>[^<>]*</answer>\s*$", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str.strip())
    return 1.0 if match_result else 0.0
    
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score



def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()



def extract_explanation(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<reason>(.*?)</reason>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags




def query_response(input_prompt, client):

    completion = client.chat.completions.create(
        model="llama3-8b-instruct",
        messages=[{"role": "user", "content": input_prompt}])

    return completion.choices[0].message.content




def reason_reward(solution_str, ground_truth, client, extra_info):


    doc = extra_info['doc']
    claim = extra_info['claim']
    ori_prompt = USER_PROMPT_ori.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)
    ori_response = query_response(ori_prompt, client).strip()
    ori_answer = extract_solution(ori_response)


    extracted_explain = extract_explanation(solution_str)
    if extracted_explain is None:
        return 0

    explain_prompt = USER_PROMPT_with_reason.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim).replace("[EXPLANATION]", extracted_explain)
    explain_response = query_response(explain_prompt, client).strip()
    explain_answer = extract_solution(explain_response)


    if explain_answer is None:
        return 0.0

    # 检查两次结果是否都正确

    if ori_answer is None:
        ori_score = 0
    else:
        ori_score = em_check(ori_answer, ground_truth)

    explain_score = em_check(explain_answer, ground_truth)


    do_print = random.randint(1, 64) == 1
    if do_print:
        print("--------------------------------")
        print(f"Reason: {extracted_explain}")
        print(f"Ans_w/o_reason: {ori_answer}")
        print(f"Ans_w_reason: {explain_answer}")

    if ori_score == 1 and explain_score == 1:
        return 1.0
    elif ori_score == 0 or explain_score == 1:
        return 1.0
    else:
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):

    answer = extract_solution(solution_str=solution_str)
    open_count, close_count = count_answer_tags(solution_str)
    do_print = random.randint(1, 64) == 1

    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # openai.base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(base_url="http://0.0.0.0:8000/v1",api_key='ssz')


    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        if answer is not None:
            print(f"Extracted answer is not None: {answer}")
        else:
            print("Extracted answer: None!")
        print(f"Solution string: {solution_str}")

    format_score = format_reward(solution_str)
    if answer is None:
        em_score = 0.0
    else:
        if em_check(answer, ground_truth):
            if open_count > 10 or close_count > 10:  # prevent output a lot of </answer>
                em_score = 1 / 4
            else:
                em_score = 1
        else:
            em_score = 0
    
    reason_score = reason_reward(solution_str, ground_truth, client, extra_info)
    final_reward = em_score + format_score + reason_score
    return final_reward
