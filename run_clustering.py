import scipy, os
import numpy as np
from tqdm import tqdm
from utils_io import load_pickle, write_pickle,load_transformer, load
from utils_similarity import train_load_embeddings, score_phr, generate_input_batch
from sklearn.metrics.pairwise import cosine_similarity
import sys
import warnings

warnings.filterwarnings("ignore")


def extract_scores(corr_type, event_list, cause_effect, file_path, root='data'):
    embs = train_load_embeddings(f'{root}/embeddings.npy', event_list)
    N = embs.shape[0]
    file_path = f'{root}/{file_path}'

    m = scipy.sparse.lil_matrix((N, N))
    if corr_type == 'phr':
        print('Loading paraphrasing model ...')
        
        model, tokenizer = load_transformer('phr')
        
        file = open(file_path, 'w+') 
    
    
    for i in tqdm(range(N)):
        sims = cosine_similarity(embs[i:i+1], embs)[0, ]
        if cause_effect is not None:
            ce = cause_effect[i]['causes'].union(cause_effect[i]['effects'])
        if corr_type == 'phr':
            locs = np.argwhere(sims >= 0.80)[:, 0].tolist() 
        else:
            locs = np.argwhere(sims > 0.60)[:, 0].tolist()    
        
        if cause_effect is not None: locs = [i for i in locs if i not in ce]
        if corr_type == 'phr':
            probs = []
            for b in range(0, len(locs), 10):
                batch_locs = locs[b: b + 10]
                input = generate_input_batch(event_list, i, batch_locs, 'phr')
                pr = score_phr(input, tokenizer, model)
                probs.append(pr)

            probs = np.concatenate(probs)
            for index, l in enumerate(locs): 
                file.write(f'{i},{l},{probs[index]}\n')

        else:
            m[i, locs] = sims[locs]
    
    if corr_type == 'phr': 
        file.close()
    else:
        m = m.tocsr()
        np.save(file_path, m, allow_pickle=True)

def load_scores(corr_type, event_list, cause_effect, root='data'):
    if corr_type == 'similarity':
        file_path = f'{root}/similarity_matrix.npy'
    else: 
        file_path = f'{root}/phr.score'
    if not os.path.isfile(file_path):
        print(f'Extracting {corr_type} scores ...')
        extract_scores(corr_type, event_list, cause_effect, file_path)
    else: 
        if corr_type == 'similarity':
            print('Loading pre-trained scores ...')
            m = np.load(f'{root}/similarity_matrix.npy', allow_pickle=True)
            return m
        
        else:
            print('Loading paraphrasing scores ...')
            data = load(file_path)
            phr_dict = {}
            for line in data:
                x, y, s = line.split(',')
                x, y, s = int(x), int(y), eval(s)
                phr_dict[(x,y)] = s 
            return phr_dict 
            


def do_clustering(method, similarity_matrix, cause_effect, threshold, path):
    
    print(f'Run {method} clustering ...')    
    N = similarity_matrix.shape[0]

    X = np.array(range(N)).reshape(-1, 1)
    def distance(Xi, Xj): return 1 - similarity_matrix[int(Xi[0]), int(Xj[0])]

    if method == 'louvain':

        from sknetwork.clustering import Louvain

        louvain = Louvain()
        # Input adjacency matrix
        clustering = louvain.fit(similarity_matrix.astype('bool'))
        write_pickle(clustering.labels_, path)

    elif method == 'leiden':
        import leidenalg as la
        from igraph import Graph
        g = Graph.Weighted_Adjacency(similarity_matrix)
        partition = la.find_partition(g, la.ModularityVertexPartition)
        labels = np.array(partition.membership)
        write_pickle(labels, path)
    
    elif method == 'agglomerative':
        # input must be distance matrix
        
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(metric="precomputed",
                                                distance_threshold=th,
                                                n_clusters=None,
                                                linkage="complete")
        distance_matrix = 1 - similarity_matrix.toarray()
        clustering.fit(distance_matrix)
        write_pickle(clustering.labels_, path)
    
    elif method == 'optics':
        # input must be distance matrix
        
        from sklearn.cluster import OPTICS
        clustering = OPTICS(metric=distance)
        clustering.fit(X)
        write_pickle(clustering.labels_, path)

    elif method == 'hdbscan':
        # input must be distance matrix and symmetric
        from sklearn.cluster import HDBSCAN
        
        clustering = HDBSCAN(metric=distance)
        clustering.fit(X)
        write_pickle(clustering.labels_, path)
        

    elif method == 'manual':

        def find_neighbors(center, similarity_matrix, threshold, cause_effect):
            '''
            search for related events semantically and "causally" (if possible),
            where "causally" means sharing the same cause or effect. 
            '''
            arr = similarity_matrix[center].reshape(-1, 1) 
            nbs = np.argwhere(arr >= threshold)[:, 0].tolist()
            nbs = set(nbs)
            rcs, res = cause_effect[k]['causes'], cause_effect[k]['effects']
            updated_nbs = set()
            for cand in nbs: 
                cs, es = cause_effect[cand]['causes'], cause_effect[cand]['effects']
                if len(rcs & cs) > 0 or len(res & es) > 0 :
                    updated_nbs.add(cand)
            
            if len(updated_nbs) > 0: 
                return updated_nbs
            else:
                return nbs

        def loop_find_neighboors(center_list, similarity_matrix, threshold, cause_effect, clusters, assigned_events):
            for center in center_list: 
                if center not in assigned_events:
                    nbs = find_neighbors(center, similarity_matrix, threshold, cause_effect)
                    assigned_events.add(center)
                    # ignore events already been clustered
                    nbs = nbs - assigned_events
                    clusters[center] = nbs
                    assigned_events.update(nbs)
            return clusters, assigned_events
        
        clusters = {}
        assigned_events = set() # track events already been clustered

        for k in tqdm(cause_effect):
            causes, effects = cause_effect[k]['causes'], cause_effect[k]['effects']
            clusters, assigned_events = loop_find_neighboors([k], similarity_matrix, threshold, cause_effect, clusters, assigned_events)
            clusters, assigned_events = loop_find_neighboors(causes, similarity_matrix, threshold, cause_effect, clusters, assigned_events)
            clusters, assigned_events = loop_find_neighboors(effects, similarity_matrix, threshold, cause_effect, clusters, assigned_events)
        

        labels = np.zeros(shape=(N,)) - 1
        labels = labels.astype('int')

        cluster_index = 0
        for k, v in clusters.items():
            labels[k] = cluster_index
            for elem in v: 
                labels[elem] = cluster_index
            cluster_index += 1

        write_pickle(labels, path)
            
    else: 
        raise ValueError('Method not supported!')

if __name__ == "__main__":

    import os
    from cluster import Cluster
    from sklearn import metrics
    from utils_similarity import inter_cluster_similarity
    
    method = sys.argv[1]
    cluster_manager = Cluster()
    
    # method = list(map(str, method.strip('[]').split(',')))

    root = 'data'

    path = f'{root}/{method}.cluster'
    
    # both cosine and phr scores are symmetric
    m = load_scores('similarity', cluster_manager.event_list, cluster_manager.cause_effect, root=root)
    similarity_matrix = m.item()


    phr_dict = load_scores('phr', cluster_manager.event_list, cluster_manager.cause_effect, root=root)
    N = similarity_matrix.shape[0]
    print(N)

    th = 0.70
    weight = 0.50
    locs = np.argwhere(similarity_matrix >= th)

    final_matrix = scipy.sparse.lil_matrix((N, N))

    for x, y in tqdm(locs):  
        phrs = phr_dict[(x, y)] if (x,y) in phr_dict else 0     
        score = weight * similarity_matrix[x, y] + (1-weight) * phrs
        score = max(final_matrix[x, y], final_matrix[y, x], score)
        final_matrix[x, y] = score
        final_matrix[y, x] = score
    
    similarity_matrix = final_matrix.tocsr() 

    
    if not os.path.isfile(path):        
        do_clustering(method, similarity_matrix, cluster_manager.cause_effect, th, path)


    print('Evaluation begins ...')
    

    # path = f'data/{method}.cluster'
    cluster_manager.load_cluster(path)
    cluster_manager.evaluate()
    
    
    X = train_load_embeddings(f'{root}/embeddings.npy', cluster_manager.event_list)

    # Calinski-Harabasz index: score is higher when clusters are dense and well separated
    chi = metrics.calinski_harabasz_score(X, cluster_manager.cluster)
    print('- Calinski-Harabasz index:', chi)


    # Silhouette Coefficient
    slh = metrics.silhouette_score(X, cluster_manager.cluster, metric='euclidean')
    print('- Silhouette Coefficient (Euclidean) :', slh)
    slh = metrics.silhouette_score(X, cluster_manager.cluster, metric='cosine')
    print('- Silhouette Coefficient (Cosine) :', slh)

    # Inter cluster similarity
    inter_sim = []
    for k, cluster in tqdm(cluster_manager.collector.items()): 
        # exclude single-event cluster
        if len(cluster) > 1 and k > -1:
            sim = inter_cluster_similarity(cluster, similarity_matrix, cluster_manager.cause_effect)
            inter_sim.append(sim)
    print('- Inter cluster similarity: ', sum(inter_sim) / len(inter_sim))

    from finetune import print_cluster 
    print_cluster(cluster_manager, 'test')