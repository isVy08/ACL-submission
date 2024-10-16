import torch
import os,sys,re
import random
from tqdm import tqdm
import pandas as pd
from utils_io import load_json, write_json
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM

model_name = sys.argv[1]
if model_name == 'llama2':
    path_to_llama = f"/home/tvoo0019/pb90_scratch/vyvo/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/"
elif model_name == 'llama32': 
    path_to_llama = f"/home/tvoo0019/pb90_scratch/vyvo/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/"
else:
    path_to_llama = f"/home/tvoo0019/pb90_scratch/vyvo/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/" 

path_to_llama = path_to_llama + os.listdir(path_to_llama)[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def send_query_llama(query, model, tokenizer):
    inputs = tokenizer(query, padding=True, return_tensors="pt").to(model.device)
    generate_ids = model.generate(inputs.input_ids, do_sample=True, top_p=0.9, repetition_penalty=1.25, temperature=0.8, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id,)
    generate_response = tokenizer.decode(generate_ids[:, inputs.input_ids.shape[-1]:][0], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    return generate_response

def generate_demo(item): 
    string = f"Here is an example question: {item['question']}\n" if not causal else f"This is an example question: {item['question_wcg']}\n"
    string += 'Choose one or more correct answers out of the following choices:\n'
    for i, choice in enumerate(item['choices']):
        string += f'{i}. {choice}\n'
    answer = [] 
    for ans in item['answer']: 
        if ans in item['choices']: 
            answer.append(item['choices'].index(ans))
    string += f'The correct answer(s) are: {answer}\n'
    string += 'Now it is your turn to answer the following question.\n'
    return string

def generate_cot(question, about): 
    for sent in question.split('.'):
        if 'The story describes an event where' in sent: 
            event = sent.replace('The story describes an event where', '').strip()
            break 

    prompt = f"The event {event} is described by one of the sentences in the story context. First identify that part of the story."
    prompt += f" Then retrieve the event mentioned in the story that is a corresponding {about}."
    return prompt

def llm_for_causalReasoning(filename, causal, path_to_llama, prompt_type=None, seed=None):

    qa_dict = load_json(filename + '.json')

    if seed is not None:
        filename = filename[:16] + f'S{seed}/' + filename[16:]

    if prompt_type == 'demo': 
        filename += '-demo'
        cause_keys = [key for key in qa_dict if qa_dict[key]['about'] == 'cause']
        effect_keys = [key for key in qa_dict if qa_dict[key]['about'] == 'effect']
    
    elif prompt_type == 'cot':
        filename += '-cot'

    if causal: filename += '-wcg'
    
    print("Loading LLM model...")
    llama_model = LlamaForCausalLM.from_pretrained(path_to_llama).to(device)
    llama_tokenizer = AutoTokenizer.from_pretrained(path_to_llama)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id

    output = {}

    for key, item in tqdm(qa_dict.items()):
        
        string = f"Question {key}: {item['question']}\n" if not causal else f"Question {key}: {item['question_wcg']}\n"
        string += 'Choose one or more correct answers out of the following choices:\n'
        for i, choice in enumerate(item['choices']):
            string += f'{i}. {choice}\n'

        if prompt_type == 'demo': 
            prompt_key = random.choice(cause_keys) if item['about'] == 'cause' else random.choice(effect_keys)
            prompt = generate_demo(qa_dict[prompt_key])
            string = prompt + string
        elif prompt_type == 'cot':
            prompt = generate_cot(item['question'], item['about'])
            string = string + prompt

        string += "Let’s work this out in a step-by-step way to be sure that we have the right answer.\n"
        string += "Provide the answer beginning with 'The correct answer(s):' followed by a list of the indices of the correct answers. Note that there can be multiple correct answers.\n"
        
        
        msg = [{"role": "user", "content": string}]
        input_str = llama_tokenizer.apply_chat_template(msg, tokenize=False)
        response = send_query_llama(input_str, llama_model, llama_tokenizer)
        output[key] = response


    write_json(output, filename + f'-{model_name}.json')


def llm_for_causalDiscovery(read_file, save_file):
   
    df = pd.read_csv(read_file)
    # load LLM model
    llama_model = LlamaForCausalLM.from_pretrained(path_to_llama).to(device) 
    llama_tokenizer = AutoTokenizer.from_pretrained(path_to_llama)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    
    output = {}

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        cluster_a = row['Cluster A'].split(':', 1)[1].strip()
        cluster_b = row['Cluster B'].split(':', 1)[1].strip()
        query = f"Which cause-and-effect relationship is more likely between two events? A. '{cluster_a}' causes '{cluster_b}'. " \
               f"B. '{cluster_b}' causes '{cluster_a}'. C. There are no cause-effect relation between two events. Let’s work this out in a step-by-step way to be sure that we have the right answer. Then provide one final answer within the tags <Answer>A or B or C</Answer>."
    
        msg = [{"role": "user", "content": query}]
        input_str = llama_tokenizer.apply_chat_template(msg, tokenize=False)
        response = send_query_llama(input_str, llama_model, llama_tokenizer)
        output[index + 1] = response

    write_json(output, save_file)
    

task = sys.argv[2]
if task == 'cd':
    for version in ('v2', 'v3'):
        print(f'Running causal discovery at {version} for {model_name}')
        data_file = f"CAUSAL/data/test-graph-{version}.csv"
        output_file = f"CAUSAL/output/{model_name}-response-{version}.json"
        llm_for_causalDiscovery(data_file, output_file)

else: 
    _name_dict = {'GQSA': 'v1', 'SQSA': 'v1'}
    prompt_type = 'demo'
    for name, ver in _name_dict.items():
        for causal in (True,False):
            print('Running QA:', name, 'causal =', causal)
            filename = f'GLUCOSE-QA/{name}/mc-{name}-{ver}'
            llm_for_causalReasoning(filename, causal, path_to_llama, prompt_type, seed=1)
