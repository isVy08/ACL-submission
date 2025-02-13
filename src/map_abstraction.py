from metadata import Glucose
from utils.causality import get_id, get_text
from metagraph import ACCESS
from tqdm import tqdm 
import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")


'''
Map abstraction to specific mentions
'''

graph = ACCESS(root='../benchmark',path_to_graph='final_graph.csv', path_to_cluster='final_cluster.csv')


db = Glucose()
N = len(db.dataset)
D = len(graph.collector)
topic_ids = list(graph.collector.keys())

def count_overlap(text_a, text_b, excluded={'person', 'a', 'another', 'something', 'someone', 'be'}):
    tokens_a = set(text_a.split(' '))
    tokens_a = tokens_a.difference(excluded)
    tokens_b = set(text_b.split(' '))
    return len(tokens_a & tokens_b)

def smoothen(sent):
    tokens = sent.split(' ')
    tokens = [tok.strip() for tok in tokens if len(tok) > 0]
    return ' '.join(tokens)

def extract_main_verb(sent):
    sent = smoothen(sent)
    doc = nlp(sent)
    for token in doc:
        if token.pos_ == 'VERB' or token.dep_ == 'ROOT':
            return token.lemma_

level1 = []
level2 = []
level3 = []
for i in tqdm(range(D)):
    tid = topic_ids[i]
    cluster = graph.collector[tid]['cluster']
    topic = graph.collector[tid]['topic']
    topic = topic.replace('topic : ', '')
    
    for sent in cluster:
        idx = int(get_id(sent))
        event = db.event_list[idx]
        orig_ids = db.updated_events[event]
        for oi in orig_ids:
            # story loc
            loc = list(db.event_loc[oi])[0]
            gen_exprs = db.dataset[loc]['1_generalNL'].split(' >Causes/Enables> ')
            spec_exprs = db.dataset[loc]['1_specificNL'].split(' >Causes/Enables> ')
            orig_event = smoothen(db.events[oi]) 
            ref0 = smoothen(spec_exprs[0])
            ref1 = smoothen(spec_exprs[1])

            root = extract_main_verb(orig_event)
            v0 = extract_main_verb(ref0)
            v1 = extract_main_verb(ref1)
            check_overlap = True if root is None else False
            if root is not None: 
                if root == v0: 
                    output = ref0
                elif root == v1:
                    output = ref1
                
                if root == v0 or root == v1:
                    level1.append(topic)
                    level2.append(get_text(sent))
                    level3.append(output) 
            

df = pd.DataFrame({'Level 1': level1, 'Level 2': level2, 'Level 3': level3})
df.to_csv('abstraction.csv', index=False)       
            




