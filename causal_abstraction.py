import spacy
from nltk.tokenize import sent_tokenize
from utils_io import load_pickle, write_pickle, write_json, load_json

import time
# from retry.api import retry_call
# import openai
import random
# import keys
import pandas as pd
from tqdm import tqdm
from utils_io import write_pickle
from datasets import load_dataset, concatenate_datasets
from nltk.translate.bleu_score import sentence_bleu

# nltk.download('punkt')

        

def remove_exclamation_words(sent):
    words = ['Hello, ', 'Well, ', 'Yes, ', 'Er, ', 
            'No, ', 'OK, ', 'Yeah, ', 'Hmm, ', "I'm sorry, but ", "Oh, ", "I'm afraid "]
    for w in words: 
        sent = sent.replace(w, '')
    return sent 

def extract_question_target():
    nlp = spacy.load("en_core_web_sm")
    dataset_split = load_dataset("declare-lab/cicero", cache_dir='data')

    # Select question 
    questions = ['What is or could be the cause of target?',
                'What subsequent event happens or could happen following the target?']

    database = {}
    dataset = concatenate_datasets([dataset_split['train'], dataset_split['test']])
    record_id = 0
    for diag in tqdm(dataset): 
        if diag['Question'] in questions: 
            target = diag['Target']
            sentences = sent_tokenize(target)
            selected = set()
            for sent in sentences:
                # exclude questions and short sentences
                if sent[-1] != '?' and len(sent.split(' ')) > 5: 
                    sent = remove_exclamation_words(sent)
                    doc = nlp(sent)
                    deps = [tok.dep_ for tok in doc]
                    # only include affirmative sentence
                    if 'nsubj' in deps: 
                        selected.add(sent)
            
            if len(selected) > 0:
                diag_id = diag['ID'] + '-' + str(record_id)
                if diag_id not in database: 
                    database[diag_id] = {}
                
                database[diag_id][diag['Question']] = selected
        record_id += 1
    # os.makedirs('data/', exist_ok=True)
    write_pickle(database, 'data/cicero.db')



# Get abstraction 

def get_response(prompt):
    chatgpt_query_time = time.time()
    # print(prompt)
    try:
        completion = retry_call(openai.ChatCompletion.create, fkwargs={"model":"gpt-3.5-turbo",
            "messages":[{"role": "user", "content": prompt}],
            "api_key":random.choice(keys.keys),
            "n":1,"temperature":0.25, "request_timeout":30}, tries=10, delay=1, jitter=1)
        # completion = openai.ChatCompletion.create(
        #    model="gpt-3.5-turbo",
        #    messages=[{"role": "user", "content": prompt}],
        #    api_key=random.choice(keys),
        #    n=4,
        #    temperature=0.25,
        # )
    except:
        print('-----------------------------openai API is failed!!!!!------------------------------------')
        completion = {'choices':[{'message':{'content':'Error'}}]}
    """
    nocontext_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": no_context_prompt}],
        api_key=random.choice(keys),
        n=3,
        temperature=0.25,
    )
    """
    print("chatgpt query time is : {}".format(str(time.time()-chatgpt_query_time)))
    return completion

def get_input(file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    list_ = []

    for (a,b,c) in zip(df['Level_1'], df['Level_2'], df['Level_3']):
        dict_ = {'Level_1':a, 'Level_2':b, 'Level_3':c}
        list_.append(dict_)
    return list_

def compare_references(refers, cands):
    if len(refers) != len(cands):
        print('ERROR! The length for references and candidates are not the same!')
    else:
        references, candidates = [], []
        correct_count = 0
        for ref, cand in zip(refers, cands):
            if ref.strip() == cand.strip():
                correct_count += 1
            # Tokenize sentences
            references.append(ref.split())
            candidates.append(cand.split())

        # Calculate BLEU for each pair
        scores = [sentence_bleu([ref], cand) for ref, cand in zip(references, candidates)]

        # Calculate average BLEU
        average_bleu = sum(scores) / len(scores)
        print(f"Average BLEU Score: {average_bleu}")
        correctness = float(correct_count / len(references))
        print(f"Perfect Match Rate: {correctness}")

def extract(content):
    lists = [x.strip() for x in content.strip().split('\n') if x.strip() != '']
    print(lists)

    level_2_result, level_1_result = "", ""
    for i, item in enumerate(lists):
        if '2. conversion' in item.lower() and i < len(lists)-1:
            level_2_result = lists[i+1]
        if 'further conversion' in item.lower() and i < len(lists)-1:
            level_1_result = lists[i+1]
    return level_2_result, level_1_result


def main():
    prompt = ""
    # Open the file in read mode
    with open('causal_prompt.txt', 'r') as file:
        # Read all lines
        lines = file.readlines()

    for line in lines:
        prompt += line
    prompt = prompt.strip()

    abstractions = {}
    database = load_pickle('data/cicero.db')

    for record_id, value in database.items():
        # print(f'-------------- Processing record {record_id} --------------')
        for question, target in value.items():
            for text in target:
                text = text.capitalize()
                temp_prompt = prompt.replace("$$$$$", text)
                completion = get_response(temp_prompt)
                content = completion['choices'][0]['message']['content']
                if content != 'Error':
                    level_2_result, level_1_result = extract(content)
                    abstractions[text] = (level_2_result, level_1_result)

    write_pickle(abstractions, 'data/cicero.abs') 

def extract(content):
    lists = [x.strip() for x in content.strip().split('\n') if x.strip() != '']
    print(lists)

    level_2_result, level_1_result = "", ""
    for i, item in enumerate(lists):
        if '2. conversion' in item.lower() and i < len(lists)-1:
            level_2_result = lists[i+1]
        if 'further conversion' in item.lower() and i < len(lists)-1:
            level_1_result = lists[i+1]
    return level_2_result, level_1_result

def retrieve_the_gold(input_, ref_list):
    THRESHOLD = 0.7

    df = pd.read_csv('causal_abstraction_output.csv')
    # ref1 = list(set(df['Ref_Level_1'].tolist()))

    prompt = ""
    # Open the file in read mode
    with open('causal_prompt.txt', 'r') as file:
        # Read all lines
        lines = file.readlines()
    for line in lines:
        prompt += line
    prompt = prompt.strip()
    temp_prompt = prompt.replace("$$$$$", input_)
    completion = get_response(temp_prompt)
    content = completion['choices'][0]['message']['content']
    max_sim = (-1, '')
    if content != 'Error':
        level_2_result, level_1_result = extract(content)
        if level_1_result != '':
            abstract = level_1_result
            for ref in ref_list:
                sim_ = sentence_bleu([ref.split()], abstract.split())
                max_sim = (sim_, ref) if max_sim[0] < sim_ else max_sim
            max_sim = max_sim if max_sim[0] > THRESHOLD else (-1, '')
    return max_sim

# ========================= EXPERIMENT STARTS HERE =========================

def combine_abstraction_with_cicero_question():
    THRESHOLD = 0.7
    causal_graph_path = 'benchmark_v1/final_graph.csv'
    # Read the CSV file into a DataFrame
    df = pd.read_csv(causal_graph_path, sep=',')
    list_ = []
    for Cluster_A,Cluster_B, Relation in zip(df['Cluster A'], df['Cluster B'], df['Relation']):
        list_.append([Cluster_A, Cluster_B, Relation])
    extraction_res = load_json('cicero/cicero_abstraction_with_diag_id.json')

    res_ = []
    count = 0
    len_ = len(extraction_res)
    for dict_ in extraction_res:
        if count % 10 == 0:
            print('{} in {}!'.format(count, len_))
        count += 1
        max_sim = (0.0, None)
        abstraction_level_1 = dict_['abstraction_level_1']
        for item in list_:
            Cluster_A, Cluster_B = item[0].split('topic :')[1].strip(), item[1].split('topic :')[1].strip()
            sim_ = sentence_bleu([abstraction_level_1.split()], Cluster_A.split())
            max_sim = (sim_, item) if max_sim[0] < sim_ else max_sim
            sim_ = sentence_bleu([abstraction_level_1.split()], Cluster_B.split())
            max_sim = (sim_, item) if max_sim[0] < sim_ else max_sim
        if max_sim[0] > THRESHOLD:
            dict_['causal_graph'] = max_sim
            res_.append(dict_)
    write_json(res_, 'cicero/cicero_abstraction_with_causal_graph_extra.json')

def expand_cicero_data_with_causal_graph():
    dataset_split = load_dataset("declare-lab/cicero", cache_dir='data')
    dataset = concatenate_datasets([dataset_split['train'], dataset_split['test']])
    data = load_json('cicero/cicero_abstraction_with_causal_graph_extra.json')
    final = []
    for item in data: 
        qid = item['question_id']
        q = item['question']
        t = item['answer']
        filtered = dataset.filter(lambda x: x['ID'] == qid and x['Question'] == q and t in x['Target'])
        for row in filtered:
            new_item = item.copy()
            new_item['target'] = row['Target']
            del new_item['answer']

            for key in ('Dialogue', 'Choices', 'Human Written Answer', 'Correct Answers'): 
                new_item[key] = row[key]
            final.append(new_item)
    
    print('Number of samples:', len(final))
    write_json(final, 'cicero/cicero_abstraction_with_final_graph_test_expansion_extra.json')
    


if __name__ == "__main__":
    # main()
    # extract_question_target()
    # combine_abstraction_with_cicero_question()
    expand_cicero_data_with_causal_graph()