import random
import os, re
import pandas as pd
import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import defaultdict





'''
Meta information of the ground-truth causal graph
'''

class ACCESS: 
    
    def __init__(self, root='../benchmark/', path_to_graph='final_graph.csv',
                 path_to_cluster='final_cluster.csv'):
        
        self.collector = None 
        self.topic_ids = None
        if path_to_cluster is not None:
            df = pd.read_csv(os.path.join(root, path_to_cluster))
            collector = {}
            for row in df.iterrows():
                topic_index = row[1]['Topic Index']
                topic = row[1]['Topic']
                if topic_index not in collector:
                    collector[topic_index] = {'topic': topic, 'cluster': []}
                collector[topic_index]['cluster'].append(row[1]['Events'])

            self.collector = collector
            self.topic_ids = list(self.collector.keys())

        # Load the ground-truth graph:
        df = pd.read_csv(os.path.join(root, path_to_graph))

        # Convert the dataframe to adjacency matrix
        self.cause_effect_list = set()
        nodes = set() 
        abstractions = {}
        self.occurences = []

        for row in df.iterrows():
            node_i, node_j, relation = row[1]
            # print(node_i, node_j)
            topic_i = self.get_text(node_i)
            topic_j = self.get_text(node_j)
            
            node_i = int(self.get_id(node_i))
            node_j = int(self.get_id(node_j))

            abstractions[node_i] = topic_i
            abstractions[node_j] = topic_j
            
            
            nodes.add(node_i)
            nodes.add(node_j)
            if relation == 'C':
                self.cause_effect_list.add((node_i, node_j))
            elif relation == 'E':
                self.cause_effect_list.add((node_j, node_i))
            self.occurences.append(node_i)
            self.occurences.append(node_j)

        self.cause_effect_list = list(self.cause_effect_list)
        self.nodes = list(nodes)
        self.A = self.to_adj()
        self.G = nx.DiGraph(self.A)
        print('Is the graph a DAG?', nx.is_directed_acyclic_graph(self.G))
        # nx.find_cycle(G, orientation="original")
        if self.collector is None: 
            self.abstractions = [abstractions[node] for node in self.nodes]
        else:
            self.abstractions = [self.collector[node]['topic'].replace('topic : ', '') for node in self.nodes]

    def generate_data_matrix(self):
        '''
        Merge similar topics 
        '''

        from metadata import Glucose
        from utils_causality import return_story_loc

        self.db = Glucose()
        N = len(self.db.dataset)
        self.X = np.zeros((N, len(self.nodes)))
        count_events = 0

        for i, tid in enumerate(self.nodes):
            cluster = self.collector[tid]['cluster']
            for sent in cluster:
                count_events += 1
                locs = return_story_loc(self.db, sent)
                for l in locs:
                    self.X[l, i] += 1
        print('Total number of events:', count_events)
        print('Total number of stories:', self.X.shape[0])
    
    def get_causal_context(self, node_pair=None, topic_pair=None):
        if node_pair is not None:
            node_a, node_b = node_pair
            a, b = self.nodes.index(node_a), self.nodes.index(node_b) 
        else: 
            assert topic_pair is not None
            topic_a, topic_b = topic_pair
            a = self.abstractions.index(topic_a)
            b = self.abstractions.index(topic_b)
        story_a = set(np.argwhere(self.X[:, a])[:, 0])
        story_b = set(np.argwhere(self.X[:, b])[:, 0])
        common = list(story_a & story_b)[:2]        
        contexts = []
        for i, loc in enumerate(common):
            loc = int(loc)
            text = self.db.dataset[loc]['story']
            contexts.append(text)
        return contexts, common

    def get_cause_effect(self, topic):
        '''
        returns the causes and effects of a natural language topic/abstraction
        ''' 

        t = self.abstractions.index(topic)
        causes = np.argwhere(self.A[:, t])[:, 0].tolist()
        causes = [self.abstractions[i] for i in causes]
        effects = np.argwhere(self.A[t, :])[:, 0].tolist()
        effects = [self.abstractions[i] for i in effects]
        return {'causes': causes, 'effects': effects}

    def get_id(self, sent):
        p = re.compile(r'([0-9]+)+')
        i = p.search(sent).group(0)
        return i

    def get_text(self, sent):
        _, text = sent.split(':')
        return text.strip()
    
    def to_adj(self):
            
        '''
        nodes: list of ordered cluster id
        returns adj matrix by the order of nodes
        '''
        d = len(self.nodes)
        
        A = np.zeros(shape=(d,d))
        for node_i, node_j in self.cause_effect_list:
            # i >> j
            i = self.nodes.index(node_i)
            j = self.nodes.index(node_j)
            A[i,j] = 1
        return A


    def find_directed_paths(self, cutoff=None):
        '''
        G : networkx graph
        '''
        
        roots = [v for v, t in self.G.in_degree().items() if t == 0]
        leaves = [v for v, t in self.G.out_degree().items() if t == 0]
        
        all_paths = []
        for root in tqdm(roots):
            for leave in leaves:
                paths = nx.all_simple_paths(self.G, root, leave, cutoff=cutoff)
                all_paths.extend(paths)

        return all_paths

    def get_subgraph_info(self):
        g = self.G.to_undirected()
        sub_graphs = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]
        
        count_dag = 0
        for node_list in sub_graphs:
            node_list = sorted(list(node_list))
            
            sub_adj = self.A[node_list, :]
            sub_adj = sub_adj[:, node_list]

            sub_dag = nx.DiGraph(sub_adj)
            
            is_dag = nx.is_directed_acyclic_graph(sub_dag)
            if is_dag: count_dag += 1
            print(f'This sub-graph has {len(node_list)} nodes and {sub_adj.sum()} edges. Is it DAG {is_dag}')
        print('==========================================================')
        print(f'There are {len(sub_graphs)} subgraphs with {count_dag} DAGs')


    def remove_cycles(self):
        
        is_dag = nx.is_directed_acyclic_graph(self.G)
        step = 0
        while not is_dag:
            step += 1
            print('Processing step', step)
            edges = nx.find_cycle(self.G, orientation="original")
            e = random.choice(range(len(edges)))
            i,j = edges[e][0], edges[e][1]
            self.A[i,j] = 0.0 # remove in adj matrix 
            self.cause_effect_list.remove((self.nodes[i], self.nodes[j]))# remove in cause effect list

            self.G = nx.DiGraph(self.A)
            is_dag = nx.is_directed_acyclic_graph(self.G)

        self.get_subgraph_info()

    def find_confounders(self):
        effect_to_cause = defaultdict(list)

        # Step 1: Create a dictionary where the key is the effect, and the value is a list of causes
        for cause, effect in self.cause_effect_list:
            effect_to_cause[effect].append(cause)

        pair_confounder_dict = {}
        D = len(self.nodes)
        for i in range(D-1):
            node_i = self.nodes[i]
            for j in range(i + 1, D): 
                node_j = self.nodes[j]
        
                # Step 2: Look up the given effect to get its list of causes
                all_node_i_causes = effect_to_cause.get(node_i, [])
                all_node_j_causes = effect_to_cause.get(node_j, [])

                # Step 3: Find the shared causes to get the list of confounders
                set1 = set(all_node_i_causes)
                set2 = set(all_node_j_causes)
                shared_elements = set1.intersection(set2)
                shared_elements = shared_elements.difference(set([node_i, node_j]))
                shared_elements_list = list(shared_elements)

                if len(shared_elements_list) > 0:
                    pair_confounder_dict[(node_i, node_j)] = shared_elements_list

        return pair_confounder_dict

    def find_colliders(self):
        cause_to_effect = defaultdict(list)

        # Step 1: Create a dictionary where the key is the cause, and the value is a list of effects
        for cause, effect in self.cause_effect_list:
            cause_to_effect[cause].append(effect)

        pair_colliders_dict = {}
        D = len(self.nodes)
        for i in range(D-1):
            node_i = self.nodes[i]
            for j in range(i + 1, D): 
                node_j = self.nodes[j]
        
                # Step 2: Look up the given cause to get its list of effects
                all_node_i_effects = cause_to_effect.get(node_i, [])
                all_node_j_effects = cause_to_effect.get(node_j, [])

                # Step 3: Find the shared effects to get the list of colliders
                set1 = set(all_node_i_effects)
                set2 = set(all_node_j_effects)
                shared_elements = set1.intersection(set2)
                shared_elements_list = list(shared_elements)

                if len(shared_elements_list) > 0:
                    pair_colliders_dict[(node_i, node_j)] = shared_elements_list

        return pair_colliders_dict
