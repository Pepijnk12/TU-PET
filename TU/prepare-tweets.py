import json
from collections import Counter

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


majority_labelled_tweets = []
# Tweets with 3, 4 or 5 annotators
for corpus in corpora:
    for tweet in corpus['Tweets']:
        tweet_label_counts = get_tweet_label_count(tweet)
        majority_label = get_majority_label(tweet_label_counts)
        if majority_label:
            majority_labelled_tweets.append({
                "tweet": tweet["tweet_text"],
                "label": majority_label
            })

possible_labels = set()
for x in majority_labelled_tweets:
    possible_labels.add(x['label'])

print(sorted(list(possible_labels)))