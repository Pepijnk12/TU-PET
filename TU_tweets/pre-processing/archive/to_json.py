import csv
import json
import os

all_tweets = []
all_tweets_single_label = []

base = "processed_unanimous"
with open(base + '/MFTC_5.csv') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:

        labels = set()
        for k, v in row.items():
            if v == '1':
                labels.add(k)
        all_tweets.append({"tweet": row['text'], "labels": list(labels)})

        if len(labels) == 1:
            all_tweets_single_label.append({"tweet": row['text'], "label": labels.pop()})

with open(base + "/tweets_single_label.json", "w+") as f:
    json.dump(all_tweets_single_label, f)

with open(base + "/tweets_all_labels.json", "w+") as f:
    json.dump(all_tweets, f)
