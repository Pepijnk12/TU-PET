import json

with open("processed_unanimous/tweets_single_label.json") as f:
    labeled_tweets = json.load(f)

LABELS = ['authority', 'betrayal', 'care', 'cheating', 'degradation', 'fairness', 'harm', 'loyalty', 'non-moral',
          'purity', 'subversion']


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


def print_tweet_counts_per_label(tweets):
    print("Tweet count labeled tweets:")
    for label, label_tweets in get_tweets_per_label(tweets).items():
        print(label, len(label_tweets))
    print("Total moral:", moral_count(tweets))


def minimal_tweet_length(tweets, length):
    return [tweet for tweet in tweets if len(tweet['tweet']) > length]


def remove_duplicates(tweets):
    texts = set()
    res_tweets = []
    for tweet in tweets:
        if tweet['tweet'] not in texts:
            res_tweets.append(tweet)
            texts.add(tweet['tweet'])
    return res_tweets


print_tweet_counts_per_label(labeled_tweets)

print()
print("With minimal tweet length")
labeled_tweets = remove_duplicates(labeled_tweets)
labeled_tweets = minimal_tweet_length(labeled_tweets, 50)
print_tweet_counts_per_label(labeled_tweets)

approx_test_size = 60
ratio_test_size = approx_test_size / moral_count(labeled_tweets)
print(ratio_test_size)
