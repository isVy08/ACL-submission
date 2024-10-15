import random
import numpy as np
from tqdm import tqdm
from utils_io import load_pickle



class Cluster(object):
  def __init__(self, path='data/glucose_updated.db'):
    self.cluster = None
    self.collector = None

    events, cause_effect = load_pickle(path)

    self.events = events 
    self.cause_effect = cause_effect
    self.event_list = [ev for ev in self.events]



  def update_cluster(self, cluster):
    
    self.cluster = cluster
    self.collector = self.group_by_cluster(self.cluster)

  def load_cluster(self, path):
    '''
    cluster : np.array, assignments of each event
    '''
    self.cluster = load_pickle(path)
    self.collector = self.group_by_cluster(self.cluster)


  def group_by_cluster(self, cluster):
    collector = {}
    for event_id, cluster_id in enumerate(cluster):
      if cluster_id not in collector:
        collector[cluster_id] = []
      
      collector[cluster_id].append(event_id)
    
    return collector

  def extract_relations(self):  
    
    num_clusters = len(self.collector)
    
    # row: how many times cluster i has a member that is a cause of any member of cluster j
    # column: how many times cluster j has a member that is an effect of any member of cluster i
    M = np.zeros(shape=(num_clusters, num_clusters)) 

    for cluster_id, cl in self.collector.items():
      for event_id in cl:
        causes = self.cause_effect[event_id]['causes']
        for cau_event_id in causes:
          parent_cluster_id = self.cluster[cau_event_id]
          if parent_cluster_id > -1:
            M[parent_cluster_id, cluster_id] += 1
    
    return M
  
  def evaluate(self):
    
    num_clusters = len(self.collector)
    M = self.extract_relations()
    sparsity = 1.0 - ( np.count_nonzero(M) / float(M.size) )
    
    # bidirectional ratio
    bdr = self.bidir_ratio(M)
    
    # self-loop ratio
    arr = [len(cl) for _, cl in self.collector.items()]
    diag_ = M.diagonal() // 2
    slr = np.divide(diag_, arr).mean()
    
    max_cluster_size = max(arr) / self.cluster.shape[0]
    outlier_clusters = [len(v) for k,v in self.collector.items() if k < 0]
    single_clusters = [k for k,v in self.collector.items() if len(v) == 1]

    print('- No. clusters        :', num_clusters)
    print('- % single clusters   :', len(single_clusters) / num_clusters)
    print('- Bidirectional Ratio :', bdr)
    print('- Self-loop Ratio     :', slr)
    print('- Sparsity            :', np.round(sparsity, 2))
    print('- % Max cluster size  :', np.round(max_cluster_size, 2))
    print('- No. outliers        :', len(outlier_clusters))
    
    if len(outlier_clusters) > 0:
      print('- Max. outliers size  :', max(outlier_clusters))
    # return M 

  def bidir_ratio(self, M):
    bdr = 0
    cnt = 0
    n = M.shape[0]
    for i in tqdm(range(n-1)):
      for j in range(i+1, n):
        ce = [M[i,j], M[j, i]]
        if max(ce) != 0:
          bdr += min(ce) / max(ce)
          cnt += 1
    
    cnt = max(1, cnt)
    return bdr/cnt


  def view_cluster(self, cluster_index, num_samples = 20):
    members = self.collector[cluster_index]
    num_samples = min(num_samples, len(members))
    random_indices = random.sample(members, k = num_samples)
    for m in random_indices:
      print(self.event_list[m])

  def reset_cluster(self, updated_collector):
    # Update cluster and its index
    updated_cluster_index = {}
    new_cluster = np.zeros_like(self.cluster) - 1
    
    for i, k in enumerate(updated_collector):
        updated_cluster_index[k] = i
        # update cluster array
        v = updated_collector[k]
        for event_id in v:
          new_cluster[event_id] = i
    
    self.cluster = new_cluster    
    self.collector = {updated_cluster_index[k] : v for k, v in updated_collector.items()}
    



    

  