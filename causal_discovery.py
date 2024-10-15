import sys, os
import numpy as np
import networkx as nx
from metagraph import ACCESS
from collections import Counter
from utils_io import load_pickle, write_pickle



'''
Load clusters and story data
'''

collector = load_pickle('benchmark/final.cluster')
topic_ids = list(collector.keys())

X = np.load('benchmark/obs.npy')


'''
Load the ground-truth graph:
''' 
graph = ACCESS()
# text = "a person buy something for another person"
# graph.get_cause_effect(text)

'''
Extract densely connected subgraph >> data
''' 
config_id = int(sys.argv[1])
thresholds = [25, 30, 35, 40, 45]

threshold = thresholds[config_id]


if not os.path.isfile(f'SL/D{config_id}/truth.pickle'):
    print('Constructing graph ...')
    counter = Counter(graph.occurences)

    selected_by_graph_id = set()
    for edge in graph.G.edges():
        cluster_a, cluster_b = graph.nodes[edge[0]], graph.nodes[edge[1]]
        topic_id_a, topic_id_b, = topic_ids.index(cluster_a), topic_ids.index(cluster_b) 
        if threshold is None:
            is_connected = is_frequent = True 
        else:
            is_connected = counter[cluster_a] > 1 and counter[cluster_b] > 1
            is_frequent = X[:, topic_id_a].sum() > threshold and X[:, topic_id_b].sum() > threshold
        if is_connected and is_frequent:
            selected_by_graph_id.add(edge[0])
            selected_by_graph_id.add(edge[1])



    selected_by_graph_id = list(selected_by_graph_id)

    # Filter 1
    true_dag = graph.A[selected_by_graph_id, :]
    true_dag = true_dag[:, selected_by_graph_id]   

    G = nx.DiGraph(true_dag)
    g = G.to_undirected()
    sub_graphs = [c for c in sorted(nx.connected_components(g), key=len, reverse=True)]

    # Filter 2 
    selected_by_graph_id = [selected_by_graph_id[i] for i in sub_graphs[0]]
    true_dag = graph.A[selected_by_graph_id, :]
    true_dag = true_dag[:, selected_by_graph_id]   

    G = nx.DiGraph(true_dag)

    print('Is DAG?', nx.is_directed_acyclic_graph(G))

    selected_by_topic_id = []
    for node_graph_idx in selected_by_graph_id: 
        cluster = graph.nodes[node_graph_idx]
        topic_id = topic_ids.index(cluster)
        selected_by_topic_id.append(topic_id)
    
    write_pickle((true_dag, selected_by_topic_id), f'SL/D{config_id}/truth.pickle')
else: 
    print('Loading graph')
    true_dag, selected_by_topic_id = load_pickle(f'SL/D{config_id}/truth.pickle')
 
print('Number of nodes:', len(selected_by_topic_id))
data = X[:, selected_by_topic_id]

indices = np.argwhere(data.sum(1))[:, 0] 
data = data[indices, ]
    

device = 'cpu'


if sys.argv[2] == 'train':
    print(f'Start causal discovery for {data.shape[1]} nodes with {data.shape[0]} samples.')
    from castle.algorithms import GAE, CORL, NotearsNonlinear, DAG_GNN, GOLEM, GES

    # print('Running GAE ...')
    # ga = GAE(input_dim = data.shape[1],
    #          hidden_dim = data.shape[1], 
    #          hidden_layers=2,
    #          epochs=5,
    #          device_type=device)
    # ga.learn(data)
    # np.save(f'SL/D{config_id}/gae.npy', ga.causal_matrix)

    print('Running NOTEARS ...')
    nt = NotearsNonlinear(max_iter=200,
                          hidden_layers=(data.shape[1], 1), 
                          w_threshold=0.1,
                          rho_max=1e+30,
                          device_type=device)
    nt.learn(data)
    np.save(f'SL/D{config_id}/notears.npy', nt.causal_matrix)

    print('Running GOLEM ...')
    gl = GOLEM(device_type=device)
    gl.learn(data)
    np.save(f'SL/D{config_id}/golem.npy', gl.causal_matrix)

    print('Running DAG-GNN ...')
    gnn = DAG_GNN(device_type=device)
    gnn.learn(data)
    np.save(f'SL/D{config_id}/gnn.npy', gnn.causal_matrix)


    print('Running CORL ...')
    corl = CORL(encoder_name='transformer',
            decoder_name='lstm',
            reward_mode='episodic',
            reward_regression_type='LR',
            batch_size=64,
            input_dim=data.shape[1],
            embed_dim=64,
            iteration=2000,
            device_type=device)


    corl.learn(data)
    np.save(f'SL/D{config_id}/corl.npy', corl.causal_matrix)


    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import chisq, gsq
        
    print('Running Chi-square independence tests ...')    
    cg = pc(data, 0.01, chisq)
    np.save(f'SL/D{config_id}/chisq.npy', cg.G.graph)


    print('Running G-square independence tests ...')
    cg = pc(data, 0.01, gsq)
    np.save(f'SL/D{config_id}/gsq.npy', cg.G.graph)

else:
    from castle.metrics import MetricsDAG

    def pc(A):
        d = A.shape[0]
        for i in range(d-1):
            for j in range(i+1, d):
                if A[i,j] == - 1 and A[j,i] == 1:
                    A[i,j] = 1
                    A[j,i] = 0
                elif A[i,j] == A[j,i] == -1:
                    A[i,j] = 1
                    A[j,i] = 1
        return A

    file = open(f'SL/D{config_id}/output.txt', 'w+')
    methods = ['gae', 'gnn', 'corl', 'notears', 'golem', 'chisq', 'gsq'] 
    # methods = ['gnn', 'notears', 'golem']
    for mth in methods:
        estimated_dag = np.load(f'SL/D{config_id}/{mth}.npy')
        if mth in ('chisq', 'gsq'):
            estimated_dag = pc(estimated_dag)
        mt = MetricsDAG(estimated_dag, true_dag)
        file.write(f'{mth.upper()}:\n')
        for k, v in mt.metrics.items():
            file.write(f'{k} : {v}\n')
        file.write('======================\n')
    file.close()
