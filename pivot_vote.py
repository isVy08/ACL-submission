from utils_io import load_pickle, write_pickle
from tqdm import tqdm
import json

# def pivot_algorithm(weights="preprocessed_data/similar_pairs.dict", index2event="preprocessed_data/glucose.db"):
def pivot_algorithm(events, similarity_matrix):
    print("run pivot algorithm")
    print("loading weights")
    # pair_weights = load_pickle(weights)
    # max_idx = 0
    # for k, v in tqdm(pair_weights.items()):
    #     if k[0] > max_idx:
    #         max_idx = k[0]
    #     if k[1] > max_idx:
    #         max_idx = k[1]

    max_idx = similarity_matrix.shape[0]

    clustering = {}
    cluster_idx = 0 # number of clusters
    placed_nodes = {}
    for i in tqdm(range(0, max_idx-1), desc='clustering'):
        if len(clustering) == 0:
            new_cluster = [i]
            for j in range(i+1, max_idx):
                if (j not in placed_nodes) and (similarity_matrix[i, j] - (1 - similarity_matrix[i, j]) > 0):
                    new_cluster.append(j)
                elif (j not in placed_nodes) and (similarity_matrix[j, i] - (1 - similarity_matrix[j, i]) > 0):
                    new_cluster.append(j)
                else:
                    continue
            clustering[cluster_idx] = new_cluster
            for idx in new_cluster:
                placed_nodes[idx] = True
            cluster_idx += 1
        else:
            if i not in placed_nodes:
                new_cluster = [i]
                for j in range(i + 1, max_idx):
                    if (j not in placed_nodes) and (similarity_matrix[i, j] - (1 - similarity_matrix[i, j]) > 0):
                        new_cluster.append(j)
                    elif (j not in placed_nodes) and (similarity_matrix[j, i] - (1 - similarity_matrix[j, i]) > 0):
                        new_cluster.append(j)
                    else:
                        continue
                clustering[cluster_idx] = new_cluster
                for idx in new_cluster:
                    placed_nodes[idx] = True
                cluster_idx += 1

    print(f"number of placed node: {len(placed_nodes)}")
    # map index to event
    # events, cause_effect, event_loc = load_pickle(index2event)
    clustering_result_text = {}
    for cluster_idx, one_cluster in clustering.items():
        clustering_result_text[cluster_idx] = []
        for idx in one_cluster:
            clustering_result_text[cluster_idx].append(events[idx])

    return clustering, clustering_result_text


# def vote_algorithm(weights="preprocessed_data/similar_pairs.dict", index2event="preprocessed_data/glucose.db"):
def vote_algorithm(events, similarity_matrix):
    print("run vote algorithm")
    print("loading weights")
    # pair_weights = load_pickle(weights)
    # max_idx = 0
    # for k, v in tqdm(pair_weights.items()):
    #     if k[0] > max_idx:
    #         max_idx = k[0]
    #     if k[1] > max_idx:
    #         max_idx = k[1]
    
    max_idx = similarity_matrix.shape[0]

    # events, cause_effect, event_loc = load_pickle(index2event)
    print(len(events), max_idx)
    clustering = {}
    if len(clustering) == 0:
        clustering[0] = [0]
    k = 1 # number of clusters created so far
    for i in tqdm(range(max_idx), desc='clustering'):
        quality = {}
        for c in range(len(clustering)):
            quality[c] = 0
            for j in clustering[c]:
                if i != j:
                    quality[c] += similarity_matrix[i, j] - (1 - similarity_matrix[i, j])
                
                else:
                    continue
        max_quality = -10000
        c_best = -1
        for cluster, cluster_quality in quality.items():
            if quality[cluster] > max_quality:
                max_quality = quality[cluster]
                c_best = cluster
        if max_quality > 0:
            clustering[c_best].append(i)
        else:
            clustering[k] = [i]
            k += 1

    # map index to event
    # events, cause_effect, event_loc = load_pickle(index2event)
    clustering_result_text = {}
    num_placed_node = 0
    for k, one_cluster in clustering.items():
        num_placed_node += len(one_cluster)
        clustering_result_text[k] = []
        for idx in one_cluster:
            clustering_result_text[k].append(events[idx])
    print(num_placed_node)

    return clustering, clustering_result_text


# def best_algorithm(weights="preprocessed_data/similar_pairs.dict", index2event="preprocessed_data/glucose.db"):
def best_algorithm(events, similarity_matrix):
    print("run best algorithm")
    print("loading weights")
    # pair_weights = load_pickle(weights)
    # max_idx = 0
    # for k, v in tqdm(pair_weights.items()):
    #     if k[0] > max_idx:
    #         max_idx = k[0]
    #     if k[1] > max_idx:
    #         max_idx = k[1]
    
    max_idx = similarity_matrix.shape[0]

    # events, cause_effect, event_loc = load_pickle(index2event)
    print(len(events), max_idx)
    clustering = {}
    if len(clustering) == 0:
        clustering[0] = [0]
    k = 1 # number of clusters created so far
    for i in tqdm(range(max_idx), desc='clustering'):
        quality = {}
        for c in range(len(clustering)):
            quality[c] = 0
            for j in clustering[c]:
                if i != j:
                    i_j_quality = similarity_matrix[i, j] - (1 - similarity_matrix[i, j])
                    quality[c] = i_j_quality if i_j_quality > quality[c] else quality[c]
                else:
                    continue
                # else:
                #     # when (i, j) is not in weights, they are dissimilar.
                #     i_j_quality = -1.0
                #     quality[c] = i_j_quality if i_j_quality > quality[c] else quality[c]
        max_quality = -10000
        c_best = -1
        for cluster, cluster_quality in quality.items():
            if quality[cluster] > max_quality:
                max_quality = quality[cluster]
                c_best = cluster
        if max_quality > 0:
            clustering[c_best].append(i)
        else:
            clustering[k] = [i]
            k += 1

    # map index to event
    # events, cause_effect, event_loc = load_pickle(index2event)
    clustering_result_text = {}
    num_placed_node = 0
    for k, one_cluster in clustering.items():
        num_placed_node += len(one_cluster)
        clustering_result_text[k] = []
        for idx in one_cluster:
            clustering_result_text[k].append(events[idx])
    print(num_placed_node)

    return clustering, clustering_result_text


def compute_distance_node2cluster(node_i, node_cluster, clustering, weights):
    costMap_node2cluster = {}
    for cluster_key, cluster_elems in clustering.items():
        if cluster_key != node_cluster and len(cluster_elems) > 1:
            costMap_node2cluster[(node_i, cluster_key)] = 0
            for node_j in cluster_elems:
                if (node_i, node_j) in weights:
                    cost_i_j = (1 - weights[(node_i, node_j)])
                elif (node_j, node_i) in weights:
                    cost_i_j = (1 - weights[(node_j, node_i)])
                else:
                    cost_i_j = 1
                costMap_node2cluster[(node_i, cluster_key)] += cost_i_j
    
    distanceMap_node2cluster = {}
    for cluster_key, cluster_elems in clustering.items():
        if cluster_key != node_cluster and len(cluster_elems) > 1:
            first_term = costMap_node2cluster[(node_i, cluster_key)] 
            second_term = 0
            for cluster_key_j, cluster_elems_j in clustering.items():
                if cluster_key_j != node_cluster and cluster_key_j != cluster_key and len(cluster_elems_j) > 1:
                    second_term += (len(cluster_elems_j) - costMap_node2cluster[(node_i, cluster_key_j)])
            distanceMap_node2cluster[(node_i, cluster_key)] = first_term + second_term
        elif cluster_key == node_cluster:
            second_term = 0
            for cluster_key_j, cluster_elems_j in clustering.items():
                if cluster_key_j != node_cluster and len(cluster_elems_j) > 1:
                    second_term += (len(cluster_elems_j) - costMap_node2cluster[(node_i, cluster_key_j)])
            distanceMap_node2cluster[(node_i, cluster_key)] = second_term
        else:
            distanceMap_node2cluster[(node_i, cluster_key)] = 100000

    return distanceMap_node2cluster
                

# def BOEM_clustering_postprocess(weights="preprocessed_data/similar_pairs.dict",
#                                 clustering_path="preprocessed_data/vote_clustering_result.json",
#                                 index2event="preprocessed_data/glucose.db"):
def BOEM_clustering_postprocess(events, pair_weights, clustering_path):
    # try to merge singleton cluster
    print("loading weights")
    # pair_weights = load_pickle(weights)
    clustering_result = json.load(open(clustering_path, "r"))
    
    for cluster_key, cluster_elems in tqdm(clustering_result.items()):
        if len(cluster_elems) == 1:
            for node in cluster_elems:
                # compute and store the distance of node to each cluster, d(node, cluster_i).
                distance_node2cluster = compute_distance_node2cluster(node, cluster_key, clustering_result, pair_weights)
                min_distance = 100000
                min_distance_cluster = -1
                for pair, distance in distance_node2cluster.items():
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_cluster = pair[1]
                print(min_distance)
                clustering_result[cluster_key].remove(node)
                clustering_result[min_distance_cluster].append(node)
                print(f"previous cluster: {cluster_key}; current cluster: {min_distance_cluster}")

    # map index to event
    # events, cause_effect, event_loc = load_pickle(index2event)
    clustering_result_text = {}
    num_placed_node = 0
    for k, one_cluster in clustering_result.items():
        num_placed_node += len(one_cluster)
        clustering_result_text[k] = []
        for idx in one_cluster:
            clustering_result_text[k].append(events[idx])

    return clustering_result, clustering_result_text


# clustering_result, clustering_result_text = best_algorithm(weights="preprocessed_data/similar_pairs.dict", index2event="preprocessed_data/glucose.db")
# json.dump(clustering_result, open("preprocessed_data/best_clustering_result.json", "w"), indent=4)
# json.dump(clustering_result_text, open("preprocessed_data/best_clustering_result_text.json", "w"), indent=4)

