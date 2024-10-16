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

def extract_cot_answer(text):
    text = text.lower()
    answers = []

    text = text.replace('correct answer:', 'correct answer(s):')
    pattern = r'correct answer(s):(.*\d+?)'
    match = re.search(pattern, text, re.DOTALL)
    
    if match: 
        string = match.group(1)
        text = string
    
    for sent in text.split('\n'): 
        sent = sent.strip()
        if 'incorrect' not in sent and 'step' not in sent and sent[:2] not in ('1.', '2.'): 
            ans = re.findall(r"(\d+)", sent)
            answers.extend(ans)
    
    answers = [int(a) for a in answers if int(a) < 6]
    return answers

def extract_qa_answer(text):
    text = text.lower()
    answers = []
    
    for sent in text.split('\n'): 
        sent = sent.strip()
        if 'incorrect' not in sent and 'step' not in sent: 
            ans = re.findall(r"(\d+)", sent)
            answers.extend(ans)
    
    answers = [int(a) for a in answers if int(a) < 6]
    return answers

def parse_llama_response(filename, causal=False): 
    responses = load_json(filename)
    for key, text in responses.items():
        kwd = f'Question {key}'
        text = text.replace(kwd, '')
        if causal: 
            responses[key] = extract_causal_answer(text)
        else: 
            if 'cot' in filename:
                responses[key] = extract_cot_answer(text)
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

def jaccard_similarity(str1, str2):
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))