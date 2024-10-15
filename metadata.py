from utils_io import load_pickle
from datasets import load_dataset, concatenate_datasets
class Glucose:
    def __init__(self):
        print('Loading Glucose dataset ...')
        self.events, _, self.event_loc = load_pickle('data/glucose.db')
        self.updated_events, self.cause_effect = load_pickle('data/glucose_updated.db')
        dataset_split = load_dataset('glucose', cache_dir='data')
        self.dataset = concatenate_datasets([dataset_split['train'], dataset_split['test']])
        self.event_list = [k for k in self.updated_events]

    
    def get_meta(self, event, event_id = None):
        '''
        returns: event id, causes, effects, original sentence, story context 
        '''
        if event_id is None:
            event_id = self.event_list.index(event)
        causes = self.cause_effect[event_id]['causes']
        effects = self.cause_effect[event_id]['effects']
        print('* Sentence:', event)
        if len(causes) > 0:
            print('* Causes:')
            for c in causes:
                print(f'- {self.event_list[c]}')

        if len(effects) > 0:
            print('* Effects:')
            for e in effects:
                print(f'- {self.event_list[e]}')
        orig_ids = self.updated_events[event]
        for i in orig_ids:
            print('* Original sentence:', self.events[i])
            print('* Story context:')
            loc = self.event_loc[i]
            print(self.dataset[loc]['story'][0]) 
            print('=' * 100)
        
if __name__ == "__main__":
    db = Glucose()
    texts = ['a person slip and fall on something near another thing',
            'a person be wash something',
            'a person notice something be miss',
            'a person offer to help another person do something',
            'another person question a person about something']
    for text in texts:
        db.get_meta(text)
