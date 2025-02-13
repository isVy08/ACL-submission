import re
from metagraph import ACCESS
from utils.io import write_json, load_json



graph = ACCESS(root='../benchmark',path_to_graph='final_graph.csv', path_to_cluster='final_cluster.csv')    
graph.generate_data_matrix()

# num_stories = len(graph.db.dataset)


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
    """
    Generate the QA item for a causal-effect event pair mentioned in a story of the GLUCOSE corpus. 
    Each QA item is a dictionary with the following structure: 
         {
            'id': story/item index,
            'context': content of the story,
            'cause': (abstraction of the cause event, 
                    generalization of the cause event, 
                    mention of the cause event, 
                    choices of candidate effects, 
                    list of correct answers), 
            'effect': (abstraction of the effect event, 
                    generalization of the effect event, 
                    mention of the effect event, 
                    choices of candidate causes, 
                    list of correct answers), 
        }
    
    """
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

        _dict = {
            'id': None,
            'context': context,
            'cause': (cause_cluster, cause_gen, cause_event, effect_choices, [0]), 
            'effect': (effect_cluster, effect_gen, effect_event, cause_choices, [0]),
        }
        
        dataset.append(_dict)
    return dataset

def write_question_bank(filename):
    """
    Construct the GLUCOSE QA Question Bank
    """
    question_bank = []
    count = 0
    for node_pair in graph.cause_effect_list:
        dataset = generate_qa_item(node_pair)
        for item in dataset:
            item['id'] = count
            question_bank.append(item)
            count += 1

    write_json(question_bank, filename + '.json')


def get_qa_item(item, ask_about, gnl_question=False, gnl_answer=False):
    """
    Generate the question and answers for each item in the question bank
    """

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

    # random.shuffle(choices) # if needed
    
    return question, answer_text, choices


def write_qa_dict(filename, gnl_question=False, gnl_answer=False):
    """
    Construct QA data for evaluating LLMs
    - gnl_question (bool): True if questions are asking about the generalization of the event mentioned; False if questions are asking about the event mention.
    - gnl_answer (bool): True if the generalized event of the correct event mentions are included the correct answer list; False if the correct answer list only contains the event mentions. 
    """
    count = 1
    _dict = {}
    # Load question bank
    QB = load_json('../benchmark/GLUCOSE-QA-Question-Bank.json')
    for item in QB:
        for ask_about in ('cause', 'effect'):
          
            if ask_about == 'cause':
                causal_info = f"A possible cause of the event '{item['effect'][0]}' is '{item['cause'][0]}''."
            else:
                causal_info = f"A possible effect of the event '{item['cause'][0]}' is '{item['effect'][0]}'."

            # Raw questions, answers and choices
            question, answer_text, choices = get_qa_item(item, ask_about, gnl_question, gnl_answer)
            
            # Questions with hints about abstract causal relations
            question_wcg = question + ' This information can help answer the question: ' + causal_info
            
            
            _dict[count] = {'id': item['id'], 'question': question, 'question_wcg': question_wcg, 
                            'answer': answer_text, 'choices': choices, 'about': ask_about}
            count += 1

    write_json(_dict, filename + '.json')