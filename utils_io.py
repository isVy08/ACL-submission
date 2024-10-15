import os
from tqdm import tqdm
import numpy as np
import torch, json, pickle


def load(datadir):
    with open(datadir, encoding='utf-8') as f:
        data = f.read().splitlines()
    return data

def write(data, savedir, mode='w'):
    f = open(savedir, mode)
    for text in data:
        f.write(text+'\n')
    f.close()


def load_pickle(datadir):
  file = open(datadir, 'rb')
  data = pickle.load(file)
  return data

def write_pickle(data, savedir):
  file = open(savedir, 'wb')
  pickle.dump(data, file)
  file.close()

def load_json(file_path):
    # Open the JSON file
    with open(file_path, 'r') as file:
        # Load the JSON data into a dictionary
        data = json.load(file)
    # Now 'data' contains the dictionary representation of the JSON file
    return data

def write_json(data, savedir):
    with open(savedir, 'w', encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)


def load_transformer(usage):
  if usage == 'phr':
    # Load paraphrase detector model
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("AMHR/adversarial-paraphrasing-detector")
    model = AutoModelForSequenceClassification.from_pretrained("AMHR/adversarial-paraphrasing-detector")
    if torch.cuda.is_available():
      model.to('cuda')
  elif usage == 'nli': 
    # Load NLI model
    # from sentence_transformers import CrossEncoder
    # model = CrossEncoder('cross-encoder/nli-roberta-base')
    # tokenizer = None
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    if torch.cuda.is_available():
      model.to('cuda')
  
  return model, tokenizer

