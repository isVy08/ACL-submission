import re, os
import random
import numpy as np
from metagraph import ACCESS
from utils_llm import parse_gpt_response, parse_llama_response
from utils_io import load, write_json, load_json
from sklearn.metrics import precision_score, recall_score, f1_score


graph = ACCESS(root='benchmark',path_to_graph='final_graph.csv', path_to_cluster='final_updated.cluster')    

graph.generate_data_matrix()
num_stories = len(graph.db.dataset)


def clean(ev):
  tokens = ev.lower().split(' ')
  final_tokens = []
  for tok in tokens: 
    tok = tok.strip()
    if len(tok) > 0:
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

def parse_structure(struc, subj):
    chunks = re.findall(r'{.*?]', struc)
    sent = ''
    for ch in chunks: 
        text, pos = ch.split('_')
        if pos == '[subject]':
            if subj is None:
                subj = re.search(r'{(.*)}', text).group(1)
                subj.strip()
                return subj
            else:
                sent += subj + ' '
        else: 
            text = re.search(r'{(.*)}', text).group(1)
            sent += text.strip() + ' '
    return sent
            

def generate_choices(context, negative_event):

    stopwords = set(['is', 'am', 'are', 'was', 'were', 'a', 'the', 'an'])
    
    sents = re.split(r'[."?!]\s*', context)
    negative_tokens = set(negative_event.lower().split(' ')) - stopwords
    choices = []
    for sent in sents: 
        tokens = set(sent.lower().split(' ')) - stopwords
        if len(sent) > 1 and len(negative_tokens & tokens)/len(negative_tokens) < 0.5:
            choices.append(sent) 
    return choices

def generate_qa_item(node_pair):
    dataset = []
    contexts, story_locs = graph.get_causal_context(node_pair=node_pair)
    c, e = graph.nodes.index(node_pair[0]), graph.nodes.index(node_pair[1])
    cause_cluster, effect_cluster = graph.abstractions[c], graph.abstractions[e] 
    for i in range(len(story_locs)):
        loc = int(story_locs[i])
        context = contexts[i]
        # check event-cluster match
        cause_event, effect_event = graph.db.dataset[loc]['1_specificNL'].split(' >Causes/Enables> ')
        cause_gen, effect_gen = graph.db.dataset[loc]['1_generalNL'].split(' >Causes/Enables> ')
        cause_event = clean(cause_event)
        effect_event = clean(effect_event)
        cause_gen = clean(cause_gen)
        effect_gen = clean(effect_gen)
        cause_choices = generate_choices(context, cause_event)
        effect_choices = generate_choices(context, effect_event) 
        # cluster, event, choices, blank answers: 0 index reserved for the correct answer 
        _dict = {
            'id': None,
            'context': context,
            'cause': (cause_cluster, cause_gen, cause_event, effect_choices, [0]), 
            'effect': (effect_cluster, effect_gen, effect_event, cause_choices, [0]),
        }
        
        dataset.append(_dict)
    return dataset

def write_question_bank():
    question_bank = []
    count = 0
    for node_pair in graph.cause_effect_list:
        dataset = generate_qa_item(node_pair)
        for item in dataset:
            item['id'] = count
            question_bank.append(item)
            count += 1

    write_json(question_bank, 'annotation/QA/question-bank.json')

def write_context():     
    count = 1
    file = open('GLUCOSE-QA/qa-context-orig.txt', 'w+')
    for node_pair in graph.cause_effect_list:
        dataset = generate_qa_item(node_pair)
        for item in dataset:
            for i in range(2):
                file.write(f'Question {count}: {item["context"]}\n')
                count += 1
    file.close()

def parse_context(filename):
    responses = load(filename)
    _dict = {}
    for res in responses:
        if len(res) > 0:
            _, i = re.match("(Question \d+:)", res).span()
            q,a = res[:i-1], res[i+1:]
            _dict[q] = a 
    return _dict


def generate_qa(item, ask_about, gnl_question=False, gnl_answer=False):

    assert ask_about in ('cause', 'effect')
    target = 'effect' if ask_about == 'cause' else 'cause'

    gen_answer = item[ask_about][1].capitalize()
    spc_answer = item[ask_about][2].capitalize()
    choices = [spc_answer] + item[target][3]
    answer = item[target][4]
    
    if gnl_answer:
        choices = [gen_answer] + choices
        answer = [0] + [i + 1 for i in answer]
              
    
    if gnl_question: 
        question = f"{item['context']}\nThe story describes an event where '{item[target][1]}'. What could be the {ask_about} of the event?"
    else:
        question = f"{item['context']}\nWhat could be the {ask_about} of the event '{item[target][2]}'?"
        
    
    answer_text = [choices[i] for i in answer]

    random.shuffle(choices)
    # must adapt the change to the answer indices if choices are randomized
    # answer = [choices.index(ans) for ans in answer_text]
    
    return question, answer_text, choices


def write_qa_dict(filename, context_dict, gnl_question, gnl_answer):
    count = 1
    _dict = {}
    # Load question bank
    QB = load_json('benchmark/GLUCOSE-QA-Question-Bank.json')
    for item in QB:
        for ask_about in ('cause', 'effect'):
            target = 'effect' if ask_about == 'cause' else 'cause'
            key = f"Question {count}"
            if key in context_dict and 0 in item[target][4] and len(item[ask_about][3]) > 1:
                item['context'] = context_dict[key]
                if ask_about == 'cause':
                    causal_info = f"A possible cause of the event '{item['effect'][0]}' is '{item['cause'][0]}''."
                else:
                    causal_info = f"A possible effect of the event '{item['cause'][0]}' is '{item['effect'][0]}'."

                question, answer_text, choices = generate_qa(item, ask_about, gnl_question, gnl_answer)
                question_wcg = question + ' This information can help answer the question: ' + causal_info
                _dict[count] = {'id': item['id'], 'question': question, 'question_wcg': question_wcg, 
                                'answer': answer_text, 'choices': choices, 'about': ask_about}
            count += 1

    write_json(_dict, filename + '.json')


def write_qa_text(filename, causal=False, seed=None): 

    qa_dict = load_json(filename + '.json')

    if seed is not None:
        filename = filename[:16] + f'S{seed}/' + filename[16:]
    
    if causal: filename += '-wcg'
    
    if os.path.isfile(filename + '-chatgpt.txt'): 
        llm_dict = parse_gpt_response(f'{filename}-chatgpt.txt')
    else:
        llm_dict = None
        file = open(filename + '-chatgpt.txt', 'w+')
        file.close()

    file = open(filename + '.txt', 'w+')
    
    for key, item in qa_dict.items():
        string = f"Question {key}: {item['question']}\n" if not causal else f"Question {key}: {item['question_wcg']}\n"
        string += 'Choose one or more correct answers out of the following choices:\n'
        for i, choice in enumerate(item['choices']):
            string += f'{i}. {choice}\n'

        if llm_dict is not None: 
            correct_answer = [item['choices'].index(ans) for ans in item['answer']]
            string += f"\n- Correct answer: {correct_answer}."
            predict_answer = llm_dict[key]
            string += f'\n- Model answer: {predict_answer}'

        file.write(string + '\n\n')

    file.close()


def evaluate_qa(filename, causal=False, model='chatgpt', seed=None): 

    qa_dict = load_json(filename + '.json')

    if seed is not None:
        filename = filename[:16] + f'S{seed}/' + filename[16:]
    
    if causal: filename += '-wcg'
    
    if model == 'chatgpt':
        llm_dict = parse_gpt_response(f'{filename}-{model}.txt')
    else:
        llm_dict = parse_llama_response(f'{filename}-{model}.json')
    
    total_acc = 0
    spc_acc = 0
    gnl_acc = 0
    f1s = []
    for key, item in qa_dict.items():

        correct_answer = [item['choices'].index(ans) for ans in item['answer']]

        predict_answer = [i for i in llm_dict[key] if i < len(item['choices'])]
        
        predict_answer_text = [item['choices'][i] for i in predict_answer] 

        if 'ORIG' in filename and int(key) > 480:
            break
         
        if ('SQGA' in filename or 'GQGA' in filename):
            gen_answer = item['answer'][0]
            spc_answer = item['answer'][1]
        else:   
            gen_answer = None
            spc_answer = item['answer'][0]

        if len(set(correct_answer) & set(predict_answer)) > 0:
            total_acc += 1
        if spc_answer in predict_answer_text:
            spc_acc += 1
        if gen_answer is not None and gen_answer in predict_answer_text:
            gnl_acc += 1
        
        f1 = evaluate_f1(item['choices'], correct_answer, predict_answer)
        f1s.append(f1)
    
    return total_acc/len(f1s), spc_acc/len(f1s), sum(f1s)/len(f1s)


def evaluate_f1(choices, correct_answer, predict_answer):
    correct_labels = [0] * len(choices)
    predict_labels = [0] * len(choices)
    for idx in correct_answer:
        correct_labels[idx] = 1
    for idx in predict_answer:
        if idx < len(predict_labels):
            predict_labels[idx] = 1
    f1 = f1_score(correct_labels, predict_labels, average='macro')
    return f1





# write_question_bank()
# write_context()

# filename = f'GLUCOSE-QA/GQSA/mc-GQSA-v1'
# write_qa_text(filename, False, seed=1)
# write_qa_text(filename, True, seed=1)
   
# _name_dict = {'SQGA': (0,1,'v1'), 'GQGA': (1,1,'v1')} 
_name_dict = {'ORIG': (0,0,'orig'), 'SQSA': (0,0,'v1'), 'GQSA': (0,1,'v1')} 
for causal in (True, False):
    for name, keys in _name_dict.items():
        filename = f'GLUCOSE-QA/{name}/mc-{name}-{keys[2]}'

        print(f"================== Running {name} with causal = {causal} ================")

        if not os.path.isfile(filename + '.json'):
            print('Writing json file ...')
            context_dict = parse_context(f'GLUCOSE-QA/qa-context-{keys[2]}.txt')
            write_qa_dict(filename, context_dict, gnl_question=bool(keys[0]), gnl_answer=bool(keys[1]))
            write_qa_text(filename, causal)
        else: 
            all_acc = []
            all_sacc = []
            all_f1 = []
            for s in range(1,4):
                acc, sacc, f1 = evaluate_qa(filename, causal, model='chatgpt', seed=s)
                all_acc.append(acc)
                all_sacc.append(sacc)
                all_f1.append(f1)
            
            
            if causal:
                print(f'$\mathbf{{{np.mean(all_acc):.3f}\pm{np.std(all_acc):.3f}}}$,$\mathbf{{{np.mean(all_f1):.3f}\pm{np.std(all_f1):.3f}}}$,$\mathbf{{{np.mean(all_sacc):.3f}\pm{np.std(all_sacc):.3f}}}$')
            else:
                print(f'${np.mean(all_acc):.3f}\pm{np.std(all_acc):.3f}$,${np.mean(all_f1):.3f}\pm{np.std(all_f1):.3f}$,${np.mean(all_sacc):.3f}\pm{np.std(all_sacc):.3f}$')
            
            # evaluate_qa(filename, causal, model='llama32')

