import json
import random
from collections import Counter

LABELS = ['authority', 'betrayal', 'care', 'cheating', 'degradation', 'fairness', 'harm', 'loyalty', 'non-moral',
          'purity', 'subversion']
MAX_UNLABELED_SIZE = 10000
MAX_VAL_SIZE = 1000


def get_tweet_label_count(tweet):
    all_tweet_labels = []
    for annotation_item in tweet['annotations']:
        labels = annotation_item['annotation'].split(',')
        for label in labels:
            all_tweet_labels.append(label)
    return dict(Counter(all_tweet_labels))


def get_majority_annotation_label(tweet):
    label_counts = get_tweet_label_count(tweet)

    sorted_label_counts = sorted(label_counts.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_label_counts) == 1:
        return sorted_label_counts[0][0]
    if sorted_label_counts[0][1] == sorted_label_counts[1][1]:
        # Equal label count so no majority vote
        return None
    return sorted_label_counts[0][0]


def get_unanimous_annotation_label(tweet):
    all_tweet_labels = set()
    for annotation_item in tweet['annotations']:
        labels = annotation_item['annotation'].split(',')
        for label in labels:
            all_tweet_labels.add(label)
    if len(all_tweet_labels) == 1:
        return all_tweet_labels.pop()
    return None


def get_annotated_tweets(tweets, vote_type):
    labeled_tweets = []
    unlabeled_tweets = []

    # Tweets with 3, 4 or 5 annotators
    for tweet in tweets:
        if vote_type == "majority":
            tweet_label = get_majority_annotation_label(tweet)
        elif vote_type == "unanimous":
            tweet_label = get_unanimous_annotation_label(tweet)
        else:
            raise Exception("Please give a valid vote_type")

        tweet_text = tweet["tweet"]

        if tweet_label:
            labeled_tweets.append({
                "tweet": tweet_text,
                "label": tweet_label
            })
        else:
            unlabeled_tweets.append({
                "tweet": tweet_text
            })

    return labeled_tweets, unlabeled_tweets


def get_tweets_per_label(tweets):
    tweets_per_label = {}
    for label in LABELS:
        tweets_per_label[label] = []

    for tweet in tweets:
        tweets_per_label[tweet['label']].append(tweet)

    return tweets_per_label


def moral_count(tweets):
    moral_count = 0
    for label, label_tweets in get_tweets_per_label(tweets).items():
        if label != 'non-moral':
            moral_count += len(label_tweets)
    return moral_count


def remove_duplicates(tweets):
    texts = set()
    res_tweets = []
    for tweet in tweets:
        if tweet['tweet'] not in texts:
            res_tweets.append(tweet)
            texts.add(tweet['tweet'])
    return res_tweets


def get_tweet_counts_per_label(tweets):
    d = {}
    for label, label_tweets in get_tweets_per_label(tweets).items():
        d[label] = len(label_tweets)
    d['total_moral'] = moral_count(tweets)
    d['total'] = len(tweets)
    return d


def print_tweet_counts_per_label(tweets):
    print("Tweet count labeled tweets:")
    for label, label_tweets in get_tweets_per_label(tweets).items():
        print(label, len(label_tweets))
    print("Total moral:", moral_count(tweets))


def minimal_tweet_length(tweets, length):
    return [tweet for tweet in tweets if len(tweet['tweet']) > length]


def store_jsonl(obj, filename):
    with open(filename, 'w+') as f:
        for tweet in obj:
            f.write(json.dumps(tweet) + "\n")


def remove_non_moral(labeled_tweets, remove_num):
    res_labeled_tweets = []
    for tweet in labeled_tweets:
        if tweet['label'] == "non-moral" and remove_num > 0:
            remove_num -= 1
        else:
            res_labeled_tweets.append(tweet)
    return res_labeled_tweets


def store_set_statistics(tweets, filename):
    with open(filename, 'w+') as f:
        json.dump(get_tweet_counts_per_label(tweets), f)


def split_val_test(labeled_tweets, unlabeled_tweets):
    random.shuffle(labeled_tweets)
    random.shuffle(unlabeled_tweets)
    unlabeled_tweets = unlabeled_tweets[:MAX_UNLABELED_SIZE]

    with open("../data/unlabeled.jsonl", 'w+') as f:
        for tweet in unlabeled_tweets:
            f.write(json.dumps(tweet) + "\n")

    print("Labeled tweets:", len(labeled_tweets))
    print("Unlabeled tweets:", len(unlabeled_tweets))

    approx_test_size = 100
    ratio_test_size = approx_test_size / len(labeled_tweets)

    tweets_per_label = get_tweets_per_label(labeled_tweets)

    train_set = []
    val_set = []

    for _, v in tweets_per_label.items():
        split_index = round(ratio_test_size * len(v))
        train_set.extend(v[:split_index])
        val_set.extend(v[split_index:])

    val_set = val_set[:MAX_VAL_SIZE]
    print("Train set count:", len(train_set))
    print("Val set count:", len(val_set))
    print("Unlabeled set count:", len(unlabeled_tweets))

    store_set_statistics(train_set, "../data/train_stats.json")
    store_set_statistics(val_set, "../data/val_stats.json")

    store_jsonl(train_set, "../data/train.jsonl")
    store_jsonl(val_set, "../data/val.jsonl")


if __name__ == '__main__':
    with open("all_tweets.json") as f:
        all_tweets = remove_duplicates(json.load(f))

    labeled_tweets, unlabeled_tweets = get_annotated_tweets(all_tweets, "unanimous")
    labeled_tweets = minimal_tweet_length(labeled_tweets, 50)
    labeled_tweets = remove_non_moral(labeled_tweets, 3400)
    print_tweet_counts_per_label(labeled_tweets)
    split_val_test(labeled_tweets, unlabeled_tweets)
