import json
import random
from collections import Counter

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


with open("pre-processing/all_tweets_cleaned.json") as f:
    in_tweets = json.load(f)

cleaned_tweets = []
for tweet in in_tweets:
    labels = []
    for annotation in tweet['annotations']:
        for label in annotation['annotation'].split(','):
            labels.append(label)
    d = dict(Counter(labels))
    max_val = d[max(d, key=d.get)]
    if max_val == 1:
        continue
    else:
        final_labels = []
        for k, v in d.items():
            if v == max_val:
                final_labels.append(k)

        cleaned_tweets.append({
            'text_a': tweet['tweet'],
            'labels': sorted(final_labels)
        })


SAMPLE_COUNT = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = PROCESSORS['tweet-task']()
wrapper = TransformerModelWrapper.from_pretrained("models/tweet-iteration3/final/p0-i0")
wrapper.model.to(device)
eval_data = [InputExample(text_a=x['text_a'], guid=random.randint(0, 10000000)) for x in cleaned_tweets][:SAMPLE_COUNT]

# print(eval_data)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(x):
  return 1/(1 + np.exp(-x))

results = wrapper.eval(eval_data, device)
LABELS = processor.get_labels()

mlb = MultiLabelBinarizer(processor.get_labels())
y_true = mlb.fit_transform([x['labels'] for x in cleaned_tweets[:SAMPLE_COUNT]])

for x in range(31):
    preds = []
    for tweet_i, logit in enumerate(results['logits']):
        threshold_indices = []
        pred = []
        for i, prob in enumerate(sigmoid(logit)):
            if prob > 0.7 + 0.01 * x:
                threshold_indices.append(i)
                pred.append(1)
            else:
                pred.append(0)
        preds.append(pred)
        # print("Pred:", ', '.join([LABELS[i] for i in threshold_indices]), " || True:", ', '.join(cleaned_tweets[tweet_i]['labels']))

    # predictions = np.argmax(results['logits'], axis=1)
    # class_idx_to_class_name = {idx: name for idx, name in enumerate(processor.get_labels())}
    # predictions = [class_idx_to_class_name[prediction] for prediction in predictions]

    print("Threshold:", round(0.7 + 0.01 * x, 2))
    print(list(zip(sklearn.metrics.f1_score(y_true.tolist(), preds, average=None), processor.get_labels())))
    print("F1", sklearn.metrics.f1_score(y_true.tolist(), preds, average='weighted'))
    print("Perfect prediction", sklearn.metrics.accuracy_score(y_true.tolist(), preds))
    print()