import random
from tqdm import tqdm
import numpy as np 
from utils_similarity import event_cluster_similarity

  
def remove_self_loop(cluster, similarity_matrix, cause_effect_dict):
  '''
  cluster : a shuffled list of members
  '''
  clusters = {}

  curr = 0
  main = []
  while len(cluster) > 0:
    x = cluster.pop()
    ce = cause_effect_dict[x]['causes'].union(cause_effect_dict[x]['effects'])
    for y in cluster:
      if y in ce :

        cluster.remove(y)
        
        if curr not in clusters: 
          clusters[curr] = {y}
        
        else:
          # check correlation with all clusters from the begining:
          scores = []
          for c in range(curr + 1):
            current_cluster = clusters[c]
            scr = event_cluster_similarity(current_cluster, y, 
                                    similarity_matrix, cause_effect_dict)
            scores.append(scr)
          
          best_score = np.max(scores)

          if best_score > 0: 
            best_cluster = np.argmax(scores)
            clusters[best_cluster].add(y)
          
          else: 
            curr += 1 
            clusters[curr] = {y}
    
    main.append(x)
  
  clusters[-1] = main
  return clusters


def search_candidates(target_cluster, ref_cluster, 
                      search_type, cause_effect_dict):
    
    removed = set()
    assert search_type in ('causes', 'effects'), "causes or effects only!"
  
    # Search member of reference_cluster is a cause/effect of any member of target_cluster
    for k in ref_cluster:
        candidates = cause_effect_dict[k][search_type] 
        for v in candidates: 
            if v in target_cluster:
                removed.add(v)
    
    return removed

def tune(cluster_manager, similarity_matrix, inter = True, intra = True):

  num_events = len(cluster_manager.events)
  
  if inter:
    print('Removing inter-cluster causal relations')
    clusters = []
    for k, cluster in cluster_manager.collector.items():
      # print('Processing cluster: ', k)
      shuffled_cluster = cluster.copy()
      random.shuffle(shuffled_cluster)
      new_cluster = remove_self_loop(shuffled_cluster, 
                                    similarity_matrix, cluster_manager.cause_effect)
      assert np.sum([len(v) for k, v in new_cluster.items()]) == len(cluster)
      v = list(new_cluster.values())
      clusters.extend(v)


    expanded_cluster = {i: None for i in range(num_events)}
    for cluster_id, cl in enumerate(clusters):
      for event_id in cl:
        expanded_cluster[event_id] = cluster_id

    updated_cluster = np.array(list(expanded_cluster.values()))
    cluster_manager.update_cluster(updated_cluster)
  
  if intra: 
    print('Removing intra-cluster causal relations')

    M = cluster_manager.extract_relations()
    num_clusters = M.shape[0]

    # Sort by the number of members in a cluster
    sorted_collector = {}
    for k, v in cluster_manager.collector.items():
      sorted_collector[k] = len(v)

      
    sorted_collector = sorted(sorted_collector.items(), key=lambda kv: kv[1], reverse=True)

    updated_collector = {}
    for i, _ in tqdm(sorted_collector):
      updated_collector[i] = [cluster_manager.collector[i]]
      for j in range(num_clusters):
        if i != j and j not in updated_collector:
          
          if min(M[i, j], M[j,i]) > 0:

            ref_cluster = cluster_manager.collector[j]
            new_cluster = []
            for idx, target_cluster in enumerate(updated_collector[i]):
                
              if M[i, j] > M[j, i]:
                # Search member A is a cause of any member B
                removed = search_candidates(target_cluster, ref_cluster, 
                                            'causes', 
                                            cluster_manager.cause_effect)
              
              else:
                # Search member A is a effect of any member B
                removed = search_candidates(target_cluster, ref_cluster, 
                                            'effects',
                                            cluster_manager.cause_effect)
              
              main = set(target_cluster)
              curr = main - removed
              if len(curr) > 0:
                new_cluster.append(curr)
              if len(removed) > 0:
                new_cluster.append(removed)
              
            
            updated_collector[i] = new_cluster

    c = 0
    for k, v in updated_collector.items():
      cluster_manager.collector[k] = v[0]
      for cl in v[1:]:
        idx = num_clusters + c
        cluster_manager.collector[idx] = cl
        c += 1 
    
    # Already update collector here
    expanded_cluster = {i: None for i in range(num_events)}
    for cluster_id, cl in cluster_manager.collector.items():
      for event_id in cl:
        expanded_cluster[event_id] = cluster_id
    

    updated_cluster = np.array(list(expanded_cluster.values()))
    cluster_manager.cluster = updated_cluster
  
def print_cluster(cluster_manager, method):
    single_cluster_count = 0
    # write_result
    file = open(f'{method}.txt', 'w+')
    sizes = []
    iter = range(len(cluster_manager.collector))
    for i in tqdm(iter):    
        cluster = cluster_manager.collector[i]
        sizes.append(len(cluster))
        if len(cluster) == 1: 
            single_cluster_count += 1
        else:
            file.write(f'Cluster {i} : {len(cluster)} members\n')            
            for c in cluster:
                # Append zeros 
                id = "X" * (5 - len(str(c))) + str(c)
                event = cluster_manager.event_list[c] 
                
                msg = f'({id}) {event}'
                file.write(msg + '\n')
            
            file.write('\n')

    file.close()

    print('- Number of single cluster   : ', single_cluster_count, '/', len(cluster_manager.collector))
    print('- Total event mentions       : ', np.sum(sizes))
    print('- Largest cluster size       : ', np.max(sizes))
    print('- Smallest cluster size       : ', np.min(sizes))
    cluster_manager.evaluate()

def null_handler(cluster_manager, null_clusters, action):

  # Option 1: Remove them
  if action == 'remove':
    for k in null_clusters: 
       del cluster_manager.collector[k]

    # Option 2: Merge cluster with zero causal relations with the largest / based on certain threshold
  elif action == 'merge':
    for cluster_index in tqdm(null_clusters):
      max_sim, max_k = 0, None
      cluster = cluster_manager.collector[cluster_index]
      for k in cluster_manager.collector: 
        if k not in null_clusters and cluster_index != k: 
          if len(cluster) < len(cluster_manager.collector[k]):
            sim = intra_cluster_similarity(cluster, cluster_manager.collector[k], similarity_matrix, cluster_manager.cause_effect)
          else:
            sim = intra_cluster_similarity(cluster_manager.collector[k], cluster, similarity_matrix, cluster_manager.cause_effect)
            if sim > max_sim:
                max_sim = sim
                max_k = k
          
      if max_sim > 0.70:
          # start merging
          cluster_manager.collector[max_k] =  list(cluster_manager.collector[max_k]) +  list(cluster)
          del cluster_manager.collector[cluster_index]
  cluster_manager.reset_cluster(cluster_manager.collector)
  return cluster_manager
  

if __name__ == "__main__":
  import os
  from utils_io import write_pickle, load_pickle
  from cluster import Cluster


  matrix_path = 'data/similarity_matrix.npy'
  m = np.load(matrix_path, allow_pickle=True)
  similarity_matrix = m.item()

  method = 'manual'
  action = 'merge'

  path = f'data/{method}.cluster'
  cluster_manager = Cluster()
  cluster_manager.load_cluster(path)

  from utils_similarity import intra_cluster_similarity, event_cluster_similarity

  # print('Initial Evaluation:')
  # cluster_manager.evaluate()

  if not os.path.isfile(f'data/{method}_tuned.cluster'):
    print('========= FINE-TUNING =========')
    tune(cluster_manager, similarity_matrix, inter = True, intra = True)
    write_pickle(cluster_manager.cluster, f'data/{method}_tuned.cluster')
  else: 
     labels = load_pickle(f'data/{method}_tuned.cluster')
     cluster_manager.update_cluster(labels)
  

  threshold = 10
  updated_collector = {}

  for k in tqdm(cluster_manager.collector):
      cluster = cluster_manager.collector[k] 
      if len(cluster) > threshold:
          # remove clusters with fewer than 10 members event 
          updated_collector[k] = cluster
              
  cluster_manager.reset_cluster(updated_collector)

  # Detect cluster with zero causal relations:
  M = cluster_manager.extract_relations()
  count_cause = set(np.argwhere(M.sum(0) == 0)[:, 0])
  count_effect = set(np.argwhere(M.sum(1) == 0)[:, 0])

  null_clusters = count_cause & count_effect
  cluster_manager = null_handler(cluster_manager, null_clusters, action)
  
  cluster_manager.evaluate()

  # how many documents a cluster is mentioned (total documents from its event mentions)
  from metadata import Glucose
  db = Glucose()

  def get_doc_ids(cl):
    all_ids = set()
    for event_id in cl:
        event = db.event_list[event_id]
        orig_ids = db.updated_events[event]
        for i in orig_ids:
          loc = db.event_loc[i]
          all_ids.update(loc)
    return all_ids
  
  # how many documents that a cause-effect pair co-occur in; remove those with freq < 2
  M = cluster_manager.extract_relations()
  N = M.shape[0]

  null_clusters = set()
  counts = []

  for i in range(N-1):
      for j in range(i + 1, N):
        if M[i,j] > 0:
          cdoc = get_doc_ids(cluster_manager.collector[i])
          edoc = get_doc_ids(cluster_manager.collector[j])
          counts.append(len(cdoc & edoc))
          if len(cdoc & edoc) < 1: 
             null_clusters.update([i,j])

  if len(null_clusters) > 0:
    cluster_manager = null_handler(cluster_manager, null_clusters, action)

  print_cluster(cluster_manager, 'manual')
  write_pickle(cluster_manager.cluster, f'data/{method}_final.cluster')

    