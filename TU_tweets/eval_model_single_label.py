import json
import random
from collections import Counter

import numpy
import sklearn
import torch
import numpy as np

# Leave this here
from sklearn.preprocessing import MultiLabelBinarizer

import TU_tweets.tweet_task_pvp
import TU_tweets.tweet_task_processor
from pet import InputExample
from pet.tasks import PROCESSORS
from pet.wrapper import TransformerModelWrapper


in_tweets = []
with open("eval-data/val.jsonl") as f:
    for line in f:
        in_tweets.append(json.loads(line))

cleaned_tweets = []
for tweet in in_tweets:
    cleaned_tweets.append({
        'text_a': tweet['tweet'],
        'labels': tweet['label']
    })


device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = PROCESSORS['tweet-task']()
wrapper = TransformerModelWrapper.from_pretrained("models/tweet-iteration3/final/p0-i0")
wrapper.model.to(device)
eval_data = [InputExample(text_a=x['text_a'], guid=random.randint(0, 10000000)) for x in cleaned_tweets]

# print(eval_data)
print(cleaned_tweets)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(x):
  return 1/(1 + np.exp(-x))

results = wrapper.eval(eval_data, device)

LABELS = processor.get_labels()

y_true = [LABELS.index(x['labels']) for x in cleaned_tweets]
predictions = [numpy.argmax(x) for x in results['logits']]

# print("F1", sklearn.metrics.f1_score(predictions, y_true, labels=LABELS, average='weighted', zero_division=0))
print("F1", sklearn.metrics.f1_score(predictions, y_true, average='macro'))
