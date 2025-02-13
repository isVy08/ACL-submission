import re 
import numpy as np
from utils.io import load_pickle

def get_id(sent):
    p = re.compile(r'([0-9]+)+')
    i = p.search(sent).group(0)
    return i

def get_text(sent, start = 8):
    return sent[start:]


def return_story_loc(db, sent):
    idx = int(get_id(sent))
    event = db.event_list[idx]
    orig_ids = db.updated_events[event]
    locs = []
    for oi in orig_ids:
        loc = db.event_loc[oi]
        for l in loc:
            locs.append(l)
    return locs

def get_edges(method):
    graph = load_pickle(f'data/{method}.graph')
    D = graph.shape[0]
    edges = []
    for i in range(D-1):
        for j in range(i+1, D):
            # i --> j: graph[i,j] == -1 and graph[j,i] == 1
            # i -- j: graph[i,j] == graph[j,i] == -1
            # i <--> j: graph[i,j] == graph[j,i] == 1

            if graph[i,j] != 0 or graph[j,i] != 0:
                edges.append((i,j))
    return edges

def get_relation(db, A, B):
    '''
    Extract causal relations: for every pair (i,j), there exists 4 types of relations
        * C: i --> j
        * E: i <-- j
        * B: i <--> j
        * NA: i and j not related
    '''
    causes = 0 
    effects = 0
    A = set([int(get_id(sent)) for sent in A])
    B = set([int(get_id(sent)) for sent in B])
    for i in A:
        C = db.cause_effect[i]['causes']
        E = db.cause_effect[i]['effects']
        if len(C & B) > 0: 
            causes +=1 
        if len(E & B) > 0:
            effects +=1 
    
    if causes > 0 and effects == 0: 
        return 'E'
    elif causes == 0 and effects > 0:
        return 'C'

    elif causes == 0 and effects == 0:
        return 'NA' 

    return 'B'

def write_graph(edge_relation_list, filepath, topics):
    file =  open(filepath, 'w+')
    file.write('Cluster A,Cluster B,Relation\n')
    for i, j, relation in edge_relation_list:
        if topics is None:
            file.write(f'{i},{j},{relation}\n')
        else:
            file.write(f'({i}) {topics[i]},({j}) {topics[j]},{relation}\n')
    file.close()

def extract_sub_graph(pivot, size, A):
    '''
    by matrix index
    '''
    def get_neighbor(i, A):
        return np.argwhere(A[i, :])[:, 0].tolist() + np.argwhere(A[:, i])[:, 0].tolist()

    def run(search_list, checker):
        candidates = set()
        if len(checker) < size:  
            for i in search_list: 
                nbs = get_neighbor(i, A)
                if len(nbs) > 1:
                    checker.add(i)
                    candidates.update(nbs)
            
        return candidates, checker
    checker = set()
    search_list = [pivot]
    while True: 
        search_list, checker = run(search_list, checker)
        if len(checker) >= size:
            break 
    return checker
