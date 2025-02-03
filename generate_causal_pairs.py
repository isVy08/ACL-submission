import random
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def construct_pairs(save_file):
    df = pd.read_csv("benchmark/final_graph.csv")
    clusters = []
    annotated_cause_effect_pairs = {}
    for _, row in df.iterrows():
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

