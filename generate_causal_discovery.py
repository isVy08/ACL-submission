import os
import random
import pandas as pd
from tqdm import tqdm
from utils_io import load
from utils_llm import parse_gpt_response, parse_llama_response
from sklearn.metrics import precision_score, recall_score, f1_score

def construct_pairs(save_file):
    df = pd.read_csv("benchmark_v1/final_graph.csv")
    clusters = []
    annotated_cause_effect_pairs = {}
    for index, row in df.iterrows():
        clusters.append(row["Cluster A"])
        clusters.append(row["Cluster B"])
        annotated_cause_effect_pairs[(row["Cluster A"], row["Cluster B"])] = row["Relation"]
    clusters = list(set(clusters))
    print(len(clusters))

    clusters_ = clusters.copy()
    random.shuffle(clusters)
    random.shuffle(clusters_)
    count = 0
    for i in range(len(clusters)):
        cluster_i = clusters[i]
        cluster_j = clusters_[i]
        if cluster_i != cluster_j:
            if ( (cluster_i, cluster_j) not in annotated_cause_effect_pairs ) and ( (cluster_j, cluster_i) not in annotated_cause_effect_pairs ):
                annotated_cause_effect_pairs[(cluster_i, cluster_j)] = "X"
                count += 1

    all_possible_cause_effect_pairs = []
    for key, value in annotated_cause_effect_pairs.items():
        all_possible_cause_effect_pairs.append([key[0], key[1], value])
    print(len(all_possible_cause_effect_pairs))
    df = pd.DataFrame(all_possible_cause_effect_pairs, columns=['Cluster A', 'Cluster B', 'Relation'])
    df.to_csv(save_file, index=False)


def chatgpt_for_causaldiscovery(read_file, save_file):
   
    df = pd.read_csv(read_file)

    file = open(save_file, 'w+')
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        cluster_a = row['Cluster A'].split(':', 1)[1].strip()
        cluster_b = row['Cluster B'].split(':', 1)[1].strip()
        query = f"Question {index+1}: Which cause-and-effect relationship is more likely between two events? 1. '{cluster_a}' causes '{cluster_b}'. " \
                f"2. '{cluster_b}' causes '{cluster_a}'. 3. There are no cause-effect relation between two events. Let’s work this out in a step by step way to be sure that we have the right answer. Then provide one final answer within the tags <Answer>1 or 2 or 3</Answer>."

        file.write(query + '\n\n')
    file.close()

def parse_gpt_response(filename):
    responses = load(filename)
    _dict = {}
    for res in responses:
        if len(res) > 0:
            q, a = res.split(': ')
            q = q.replace('Question', '').strip()
            _dict[q] = [int(i.strip()) for i in a.split(',')] 
    return _dict

def evaluate_ce_relation(data_file, llm_dict): 
    df = pd.read_csv(data_file)
    
    labels = ['C', 'E', 'X']
    true_labels = []
    pred_labels = []
    for key, value in llm_dict.items():        
        label = df.iloc[int(key)-1, 2].upper() 
        true_labels.append(labels.index(label) + 1)
        pred_labels.append(value[0])
    
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


version = 'v1'
data_file = f"CAUSAL/data/test-graph-{version}.csv"
model_name = 'llama3'


if not os.path.isfile(data_file):
    construct_pairs(save_file=data_file)

output_file = f"CAUSAL/output/{model_name}-response-{version}"
if model_name == 'chatgpt' and not os.path.isfile(output_file + '.txt'):
    file = open(output_file, 'w+')
    file.close()
    chatgpt_for_causaldiscovery(data_file, f'CAUSAL/data/chatgpt-{version}.txt')
else:
    if model_name == 'chatgpt':
        llm_dict = parse_gpt_response(output_file + '.txt')
    else: 
        llm_dict = parse_llama_response(output_file + '.json', True)
    # import pdb; pdb.set_trace()
    evaluate_ce_relation(data_file, llm_dict)
            