import json
import random
from collections import Counter

TRAIN_COUNT = 32
VAL_COUNT = 100


with open("MFTC_V4_text.json") as f:
    corpora = json.load(f)

print("Corpora:", [x['Corpus'] for x in corpora])


def get_tweet_label_count(tweet):
    all_tweet_labels = []
    for annotation_item in tweet['annotations']:
        labels = annotation_item['annotation'].split(',')
        for label in labels:
            all_tweet_labels.append(label)
    return dict(Counter(all_tweet_labels))


def get_majority_label(label_counts):
    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_label_counts) == 1:
        return sorted_label_counts[0][0]
    if sorted_label_counts[0][1] == sorted_label_counts[1][1]:
        # Equal label count so no majority vote
        return None
    return sorted_label_counts[0][0]


majority_labeled_tweets = []
unlabeled_tweets = []

# Tweets with 3, 4 or 5 annotators
for corpus in corpora:
    for tweet in corpus['Tweets']:
        tweet_label_counts = get_tweet_label_count(tweet)
        majority_label = get_majority_label(tweet_label_counts)
        tweet_text = tweet["tweet_text"].replace("\n", "")
        if majority_label:
            majority_labeled_tweets.append({
                "tweet": tweet_text,
                "label": majority_label
            })
        else:
            unlabeled_tweets.append({
                "tweet": tweet_text
            })

with open("unlabeled.jsonl", 'w+') as f:
    for tweet in unlabeled_tweets:
        f.write(json.dumps(tweet) + "\n")


LABELS = ['authority', 'betrayal', 'care', 'cheating', 'degradation', 'fairness', 'harm', 'loyalty', 'non-moral', 'purity', 'subversion']
tweets_per_label = {}
for label in LABELS:
    tweets_per_label[label] = []

for tweet in majority_labeled_tweets:
    tweets_per_label[tweet['label']].append(tweet)

print(tweets_per_label)
with open("train.jsonl", 'w+') as f:
    for tweet in majority_labeled_tweets[:TRAIN_COUNT]:
        f.write(json.dumps(tweet) + "\n")

with open("val.jsonl", 'w+') as f:
    for tweet in majority_labeled_tweets[TRAIN_COUNT:TRAIN_COUNT+VAL_COUNT]:
        f.write(json.dumps(tweet) + "\n")
