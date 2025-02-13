import numpy as np
import torch, os
from scipy import special

def event_cluster_similarity(current_cluster, candidate, 
                            similarity_matrix, cause_effect_dict):
  
  if isinstance(current_cluster, set):
    current_cluster = list(current_cluster)
  ce = cause_effect_dict[candidate]['causes'].union(cause_effect_dict[candidate]['effects'])
  ce_members = set(current_cluster) & ce

  if len(current_cluster) == 1:
    sims = similarity_matrix[candidate, current_cluster[0]]
    sims = [sims, sims]
  else:
    sims = similarity_matrix[candidate, current_cluster].toarray()[0, :]
  
  total = np.mean(sims)
    
  if len(ce_members) > 0:
    return -1 
  else: 
    return total


def inter_cluster_similarity(cluster, similarity_matrix, cause_effect_dict):
    sims = []
    for i in cluster: 
        s = event_cluster_similarity(cluster, i, similarity_matrix, cause_effect_dict)
        if s > -1:
          sims.append(s)
    
    # sims = sorted(sims, key=lambda tup: tup[1], reverse=True)
    return np.mean(sims) if len(sims) > 0 else 0


def intra_cluster_similarity(clusterA, clusterB, similarity_matrix, cause_effect_dict):
    sims = 0
    for a in clusterA: 
        s = event_cluster_similarity(clusterB, a, similarity_matrix, cause_effect_dict)
        if s == -1.0:
            # causal relation detected
            return -1.0
        else:
            sims += s
    return sims / len(clusterA)

def score_phr(batch, tokenizer, model):
  inputs = tokenizer(batch, padding=True, truncation=True)
  for k, v in inputs.items():
    inputs[k] = torch.Tensor(v).long()
    if torch.cuda.is_available():
      inputs[k] = inputs[k].to('cuda')
      
  scr = model(**inputs).logits
  scr = scr.detach().cpu().numpy()
  probs = special.softmax(scr,-1)[:, 1]
  del inputs
  return probs

def generate_input_batch(events, x, y_batch, usage):
  input = []
  for y in y_batch:
    if usage == 'nli':
      input.append((events[x], events[y]))
    elif usage == 'phr':
      s = events[x] + '. ' + events[y] + '.'
      input.append(s)
    else: 
      raise ValueError('usage takes either "phr" or "nli"')  
  return input
  

def train_load_embeddings(path, events=None):
  
  if os.path.isfile(path):
    print('Loading pre-trained embeddings ...')
    corpus_embeddings = np.load(path)

  else:
    assert events is not None, 'Event data is required!'
    print('Extracting embeddings ...')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='models')
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(events, batch_size=256, show_progress_bar=True, convert_to_tensor=False)
    print('Encoding finished!') 
    np.save(path, corpus_embeddings)

  print(corpus_embeddings.shape)
  return corpus_embeddings
