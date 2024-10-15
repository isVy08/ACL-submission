import os
import numpy as np
import pandas as pd
import krippendorff as kd
from collections import Counter
from metadata import Glucose
from statsmodels.stats import inter_rater as irr
from utils_io import load_pickle, write_pickle
from utils_causality import get_id, write_graph, return_story_loc
from annotate import collect_annotation, check_agreement

root = 'benchmark_v1'
curr_collector_path = 'benchmark/final.cluster' # base from first version
new_collector_path = 'final_updated.cluster'
clean_topic_path = 'clean_topics_v2.csv'
graph_path = 'final_graph.csv'


# ========================= ANNOTATION & COMPILING =========================

print('1. Calculating annotation agreement ...')

raters = ['Vy','OngKaiYun','Jonathan','TayHengEe']
r1 = collect_annotation(suffix_path='annotation/causal/CRC_R1_', raters = raters)
r3 = collect_annotation(suffix_path='annotation/causal/CRC_R3_', raters = raters)

# r4,5 follows new topic updated
r4 = collect_annotation(suffix_path='annotation/causal/CRC_R4_', raters = raters[:2]) 

# r5 = collect_annotation(suffix_path='annotation/causal/CRC_R5_', raters = ['T1', 'T2']) 

# Update r1 from r3
for k, v in r3.items(): r1[k] = v


# Check agreement here
mapper = {'X':0, 'C':1, 'E':2}
r1data = {k: [mapper[i.upper()] for i in v] for k, v in r1.items() if len(v) > 1}
r4data = {k: [mapper[i.upper()] for i in v] for k, v in r4.items() if len(v) > 1}
fk, kda = check_agreement(r1data)
print('Agreement on R1 data:', len(r1data), fk, kda)
fk, kda = check_agreement(r4data)
print('Agreement on R4 data:', len(r4data), fk, kda)


data = {**r1, **r4}


print("2. Extracting causal relations ...")
collector = load_pickle(curr_collector_path)
topic_ids = list(collector.keys())

relations = []
for k, l in data.items():
    counters = Counter(l).most_common()
    mc = counters[0][1]
    out = [item[0] for item in counters if item[1] == mc]
    a, b = k.split('_')
    a, b = int(a), int(b)
    a = f"({a}) {collector[a]['topic']}"
    b = f"({b}) {collector[b]['topic']}"
    if len(out) == 1: 
        if out[0] != 'X':
            relations.append((a,b,out[0]))


# Get unique nodes 
nodes = [int(get_id(edge[0])) for edge in relations] + [int(get_id(edge[1])) for edge in relations] 
nodes = list(set(nodes))
print(f"No. edges: {len(relations)}\nNo. nodes: {len(nodes)} ")


# ========================= MERGE TOPICS & CLUSTERS =========================
print('3. Merging topics and clusters ....')
df = pd.read_csv(f'{root}/{clean_topic_path}')
updated_collector = {}
mapper = {}

for row in range(df.shape[0]):
    old_id = df.iloc[row, 0]
    new_id = df.iloc[row, 2]
    new_topic = df.iloc[row, 3]
    cluster = collector[old_id]['cluster']
    mapper[old_id] = new_id
    if new_id not in updated_collector:
        updated_collector[new_id] = {
            'topic': new_topic,
            'cluster': cluster
        }
    else: 
        updated_collector[new_id]['cluster'].extend(cluster)
    
write_pickle(updated_collector, f'{root}/{new_collector_path}')

# Load new clusters and update relations
collector = load_pickle(f'{root}/{new_collector_path}')
updated_relations = []
nodes = set()
for a, b, rel in relations: 
    node_i = int(get_id(a))
    node_j = int(get_id(b))
    a = mapper[node_i]
    b = mapper[node_j]
    nodes.add(a)
    nodes.add(b)
    a = f"({a}) {collector[a]['topic']}"
    b = f"({b}) {collector[b]['topic']}"
    updated_relations.append((a,b,rel))

nodes = list(nodes)
relations = updated_relations
del updated_relations, updated_collector
print(f"No. edges: {len(relations)}\nNo. nodes: {len(nodes)} ")

# ========================= WRITE GRAPH =========================
write_graph(relations, f'{root}/{graph_path}', None)


# ========================= WRITE CLUSTER STORY DATA =========================
print('4. Writing clusters & stories ...')
db = Glucose()

file = open(f'{root}/clusters_with_stories.txt', 'w+')
cfile = open(f'{root}/clusters.txt', 'w+')

for idx, i in enumerate(nodes):
 
    cluster = collector[i]['cluster']
    stories = []
    file.write(f'Index {idx} - Cluster {i} - {collector[i]["topic"]}\n')
    cfile.write(f'Index {idx} - Cluster {i} - {collector[i]["topic"]}\n')
    for sent in cluster:
        file.write(sent+'\n')
        cfile.write(sent+'\n')
        locs = return_story_loc(db, sent)
        stories.extend(locs)
    
    file.write(f'** Stories: {str(locs)}\n')
        
    file.write('\n\n')
    cfile.write('\n\n')

file.close()
cfile.close()

print('5. Extracting causal chains ...')

from metagraph import ACCESS
graph = ACCESS(root=root, path_to_graph=graph_path, path_to_cluster=new_collector_path)


'''
Different Causal structures
'''
file = open(f'{root}/paths.txt', 'w+')



file.write('========== CONFOUNDERS ==========\n')
pair_confounder_dict = graph.find_confounders()
cnt = 0
for pair, confounder in pair_confounder_dict.items():
    i, j = pair 
    if (i,j) in graph.cause_effect_list:
        rel = '---->'
    elif (j,i) in graph.cause_effect_list:
        rel = '<----'
    else: 
        rel = None # 'xxxxx'
    if rel is not None:
        for c in confounder:
            cnt += 1
            file.write(f'Path {cnt}: [({i}) {collector[i]["topic"]} {rel} ({j}) {collector[j]["topic"]}] <--- ({c}) {collector[c]["topic"]}\n\n')

print(f'There are {cnt} confounders.')


file.write('========== MEDIATORS ==========\n')
pair_collider_dict = graph.find_colliders()

cnt = 0
for pair, collider in pair_collider_dict.items():
    i, j = pair 
    if (i,j) in graph.cause_effect_list:
        rel = '---->'
    elif (j,i) in graph.cause_effect_list:
        rel = '<----'
    else: 
        rel = None
    if rel is not None:
        for c in collider:
            cnt += 1
            file.write(f'Path {cnt}: [({i}) {collector[i]["topic"]} {rel} ({j}) {collector[j]["topic"]}] ---> ({c}) {collector[c]["topic"]}\n\n')


print(f'There are {cnt} mediators.')

file.write('========== COLLIDERS ==========\n')
cnt = 0
for pair, collider in pair_collider_dict.items():
    i, j = pair 
    if (i,j) not in graph.cause_effect_list and (j,i) not in graph.cause_effect_list:
        rel = 'xxxxx'
        for c in collider:
            cnt += 1
            file.write(f'Path {cnt}: [({i}) {collector[i]["topic"]} {rel} ({j}) {collector[j]["topic"]}] ---> ({c}) {collector[c]["topic"]}\n\n')

print(f'There are {cnt} colliders.')

# Find direct paths 
# directed_paths = graph.find_directed_paths()

# file.write('========== DIRECTED PATHS ==========\n')
# cnt = 0
# for path in directed_paths: 
#     if len(path) > 2:
#         text_path = ''
#         for count, node in enumerate(path): 
#             topic_index = graph.nodes[node]
#             topic = collector[topic_index]["topic"] 
#             topic = topic.replace('topic : ', '')
#             if count == len(path) - 1:
#                 text_path = text_path + f'{topic}.'
#             else:
#                 text_path = text_path + f'{topic} --> '
#         cnt +=1 
#         file.write(f'Path {cnt}: {text_path}\n\n')

# file.close()