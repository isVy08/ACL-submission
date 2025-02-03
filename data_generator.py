from tqdm import tqdm
from utils_io import load_pickle, write_pickle
from utils_data import flatten_text
import spacy, os


nlp = spacy.load("en_core_web_sm")


def clean_single_event(ev):
  tokens = ev.lower().split(' ')
  if tokens[-1] == '.':
    tokens = tokens[:-1]
  final_tokens = []
  for tok in tokens: 
    if tok == 'someone_a' or 'people_' in tok: 
      tok = 'a person'

    elif 'someone_' in tok: 
      tok = 'another person'  
   
    elif tok == 'something_a': 
      tok = 'something'
    
    elif 'something_' in tok: 
      tok = 'another thing'

    elif tok == 'somewhere_a': 
      tok = 'a place'

    elif 'somewhere_' in tok: 
      tok = 'another place'
    
    final_tokens.append(tok)

  final_tokens = [tok for tok in final_tokens if tok != 'some']
  ev = ' '.join(final_tokens)
  return ev

def find_cause_effect(event_id, event_list, 
                      updated_events, updated_events_inv, cause_effect): 
  event_ids = updated_events[event_list[event_id]]
  causes = set()
  effects = set()
  for i in event_ids: 
    # i is potentially a cause
    if i not in cause_effect: 
      if i in cause_effect[i + 1] and i + 1 in updated_events_inv:
          effect_event = updated_events_inv[i+1]
          effects.add(event_list.index(effect_event))
    # i is a effect
    else: 
      cause = []
      for cau in cause_effect[i]:
        if cau in updated_events_inv: 
          cause_event = updated_events_inv[cau] 
          cause.append(event_list.index(cause_event)) 
      causes.update(cause)
  
  return causes, effects  

def lemmatize(text):

  doc = nlp(text)
  clean_text = []
  for token in doc:
    lemma = token.lemma_.strip()
    if len(lemma) > 0:
      clean_text.append(lemma)
  
  return ' '.join(clean_text)

def generate_data(dataset):
  '''
  events       : set, of pre-processed text events
  cause_effect : dict, mapping an effect text event to a set of cause text event
  event_loc    : dict, mapping a text event to a set of story ids, used to trace co-occurrence
  '''

  events, event_loc = {}, {}
  cause_effect = {} # effect : cause
  i = 0
  dataset_id = 0
  for story in tqdm(dataset):
    ce = story['1_generalNL']
    if ce not in ['escaped', 'answered']:

      c, e = ce.split(' >Causes/Enables> ')
      c = lemmatize(c)
      e = lemmatize(e)

      events[i] = c
      events[i + 1] = e

      if e not in cause_effect:
        cause_effect[i + 1] = set()
      
      cause_effect[i+1].add(i)

      for j in (i, i + 1):
        if j not in event_loc: 
          event_loc[j] = set()

        event_loc[j].add(dataset_id)
      
      i += 2
    
    dataset_id += 1
  return events, cause_effect, event_loc

if __name__ == "__main__":
  
  dataset_path = 'data/glucose.db'
  if not os.path.isfile(dataset_path):

    from datasets import load_dataset, concatenate_datasets
    
    dataset_split = load_dataset('glucose', cache_dir='data')
    dataset = concatenate_datasets([dataset_split['train'], dataset_split['test']])

    events, cause_effect, event_loc = generate_data(dataset)
    write_pickle((events, cause_effect, event_loc), 'data/glucose.db')

  else:
    events, cause_effect, event_loc = load_pickle('data/glucose.db')

  updated_events = {}
  for i, ev in tqdm(events.items()):
    if '-PRON-' not in ev:
      try:
        out = flatten_text(ev)
        out = clean_single_event(out)
        if 'person' in out:
          if out not in updated_events: 
            updated_events[out] = []
          updated_events[out].append(i)
      except IndexError: 
          pass

  updated_events_inv = {}
  for ev, id_list in updated_events.items(): 
    for i in id_list: 
      updated_events_inv[i] = ev

  event_list = [k for k in updated_events]
  print('Number of events:', len(event_list))

  updated_cause_effect = {}
  for i, event in enumerate(tqdm(event_list)): 
    causes, effects = find_cause_effect(i, event_list, updated_events, 
                                      updated_events_inv, cause_effect)
    updated_cause_effect[i] = {}
    updated_cause_effect[i]['causes'] = causes
    updated_cause_effect[i]['effects'] = effects  

  write_pickle((updated_events, updated_cause_effect), 'data/glucose_updated.db')

  # updated_events is the location of the text in the original events object, 
  # then using event loc to trace back to the data in the Glucose data