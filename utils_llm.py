import re
from utils_io import load, load_json

def parse_gpt_response(filename):
    responses = load(filename)
    _dict = {}
    for res in responses:
        if len(res) > 0:
            q, a = res.split(': ')
            q = q.replace('Question', '').strip()
            _dict[q] = [int(i.strip()) for i in a.split(',')] 
    return _dict

def extract_qa_answer(text):
    
    if 'incorrect' in text:
        answers = []
        for sent in text.split('\n'): 
            if 'correct' in sent: 
                ans = re.findall(r"(\d+)", sent)
                answers.extend(ans)
    else: 
        answers = re.findall(r"(\d+)", text)
    answers = [int(i) for i in set(answers)]
    return answers

def parse_llama_response(filename, causal=False): 
    responses = load_json(filename)
    for key, text in responses.items():
        kwd = f'Question {key}'
        text = text.replace(kwd, '')
        if causal: 
            responses[key] = extract_causal_answer(text)
        else: 
            responses[key] = extract_qa_answer(text)
    return responses

def extract_causal_answer(input_string):
    answer_map = {"a": 1, "A":1, "b": 2, "B":2, "c": 3, "C": 3}
    # if input_string is not str:
    #     return "X"
    input_string = input_string.lower()
    if "<answer>" in input_string:
        start_index = input_string.rfind("<answer>") + len("<answer>")
        end_index = input_string.rfind("</answer>")
        final_answer = input_string[start_index:end_index].strip()
        if final_answer in answer_map:
            final_answer = answer_map[final_answer]
        else:
            final_answer = 3
    else:
        final_answer = 3
    return [final_answer]


def extract_qa_answer(text):
    text = text.lower()
    answers = []

    pattern = r'correct answer(.*\d+?)'
    match = re.search(pattern, text, re.DOTALL)
    
    if match: 
        string = match.group(1)
        ans = re.findall(r"(\d+)", string)
        answers.extend(ans)
        text = string
    
    for sent in text.split('\n'): 
        sent = sent.strip()
        if 'incorrect' not in sent and 'step' not in sent: 
            ans = re.findall(r"(\d+)", sent)
            answers.extend(ans)
    
    answers = [int(a) for a in answers if int(a) < 6]
    return answers