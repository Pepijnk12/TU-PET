from prepare_tweets import get_tweet_label_count, remove_duplicates, minimal_tweet_length
import json


def parse_annotators_multilabel(tweets):
    unlabeled_tweets = []
    labeled_tweets = []
    for tweet in tweets:
        labels = []
        for k, v in get_tweet_label_count(tweet).items():
            if v > 1:
                labels.append(k)

        del tweet['annotations']

        if len(labels) == 0:
            unlabeled_tweets.append(tweet)
        else:
            tweet['label'] = ','.join(sorted(labels))
            labeled_tweets.append(tweet)
    return labeled_tweets, unlabeled_tweets


if __name__ == '__main__':
    with open("all_tweets.json") as f:
        all_tweets = remove_duplicates(json.load(f))

    all_tweets = minimal_tweet_length(all_tweets, 60)
    labeled_tweets, unlabeled_tweets = parse_annotators_multilabel(all_tweets)

    count = 0
    for tweet in labeled_tweets:
        if "non-moral" in tweet['label']:
            count += 1
    print("Non-moral", count)
    print("Total labelled", len(labeled_tweets))

    with open("../multilabel_data/data/unlabeled.jsonl", 'w+') as f:
        for tweet in unlabeled_tweets:
            f.write(json.dumps(tweet) + "\n")

    TRAIN_SIZE = 100
    VAL_SIZE = 1000
    with open("../multilabel_data/data/train.jsonl", 'w+') as f:
        for tweet in labeled_tweets[:TRAIN_SIZE]:
            f.write(json.dumps(tweet) + "\n")

    with open("../multilabel_data/data/val.jsonl", 'w+') as f:
        for tweet in labeled_tweets[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]:
            f.write(json.dumps(tweet) + "\n")
