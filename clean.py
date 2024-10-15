import spacy, re
nlp = spacy.load("en_core_web_sm")

def join_chunks(chunks):
    words = []
    for ch in chunks:
        ch = re.sub(r'\(|\)', '', ch)
        tokens = ch.split(' ')
        for tok in tokens: 
            if len(tok) > 0:
                words.append(tok.strip())
    text = ' '.join(words).lower()
    return text

def extract_object(clause):
    '''
    Processing subordinate clauses in parentheses, categorized as one of 3 types:
    - Noun Phrase (NP) 
        e.g., Someone_A wants Something_A (that is an iPod) --> someone_a wants an iPod
    - Adj Phrase (AP) 
        e.g., Someone_A was making Something_A (that is edible) --> someone_a was making something_a edible
    - Verb Phrase (VP) 
        e.g., Someone_A sees Something_A (that Someone_A finds irresistible ) for sale --> someone_a sees something_a that someone_a finds irresistible for sale
    '''
    clause = re.sub(r'\(|\)', '', clause).strip()
    raw_clause = re.sub(r' is | are | am | was | were | have been ', ' be ', clause)
    content = nlp(clause)
    if 'that be ' in raw_clause or 'which be ' in raw_clause or 'who be ' in raw_clause:
        obj = content[2:].text
        if content[2].pos_ in ('NOUN', 'DET', 'PROPN') or content[2].text in ('where', 'when', 'who', 'which', 'that'): 
            return obj, 'NP'
        else:
            return obj, 'AP'
    else: 
        if content[0].pos_ in ('NOUN', 'DET', 'PROPN') and content[0].text not in ('where', 'when', 'who', 'which', 'that'): 
            return clause, 'NP'
        return clause, 'VP'

def flatten_text(text):
    '''
    text : event with '(' and ')' in the string
    '''

    chunks = re.split(r"(\(.*?\))", text)

    for i, ch in enumerate(chunks): 
        if '(' in ch and ')' in ch:
            obj, otype = extract_object(ch)
            pre = chunks[i-1].strip()
            item = pre.split(' ')[-1]    
            
            if otype == 'NP':
                # if NP: replace the object in the previous chunk with the extracted object in the subordinate clause
                tokens = obj.split(' ')
                if "'s" in pre and tokens[0] in ('a', 'an', 'the'): 
                  obj = ' '.join(tokens[1:])
                pre = pre.replace(item, '')
                chunks[i] = obj
                chunks[i-1] = pre
            else: 
                # if VP or AP: retain the object in the previous chunk and treat the extracted object as its own
                chunks[i] = obj

    output_text = join_chunks(chunks)

    return output_text