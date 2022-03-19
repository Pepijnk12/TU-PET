# Wrong pre-processed data!

## First iteration of model:
### Summary
- Accuracy validation set: 50% (not fair because predicts non-moral 90% of the time)
- 32 random training samples (most of them non-moral)
- 100 random validation samples
- 6775 unlabeled samples
- The model predicts non-moral 90% of the time
- ```roberta-base``` model

#### Compared to original model:
- Multi-label (11 labels: authority, betrayal, care, cheating, degradation, fairness, harm, loyalty, non-moral, purity, subversion)
- New dataset
- New patterns
- Same hyperparamaters

#### Possible improvements
- Use larger Roberta model (e.g. ```roberta-large```)
- Try different language models like Albert and Bert
- Change train, test and unlabeled data. Most of the training set was labeled 'non-moral'.
- Add new patterns
- Change hyper-parameters
- Remove Twitter usernames and urls from tweets

```{"train_set_before_training": 0.03125, "global_step": 5082, "average_loss": 0.07278807878396036, "train_set_after_training": 0.59375, "test_set_after_training": {"acc": 0.5}}```


###Patterns:
    p1 = text_a, . This made me feel:, self.mask], []
    p2 = [My tweet is: , text_a, . Therefore I believe in, self.mask], []
    p3 = [My tweet is: , text_a, . Therefore , self.mask, is important to me], []

- acc-p1: 0.5467 +- 0.0252
- acc-p2: 0.5367 +- 0.0153
- acc-p3: 0.49 +- 0.0265
- acc-all-p: 0.5244 +- 0.0328

### Verbalizer:
```    VERBALIZER = {
        'authority': ['authority', 'force', 'government', 'jurisdiction', 'rule'],
        'betrayal': ['betrayal', 'deception', 'dishonest', 'treason'],
        'care': ['care', 'responsibility', 'protection', 'trust'],
        'cheating': ['cheating', 'lying', 'unfair'],
        'degradation': ['degradation', 'deception', 'fraud', 'dishonest', 'dishonest'],
        'fairness': ['fairness', "equality", "equal", "fair", "justice", "honesty", "integrity", "balanced", "truth"],
        'harm': ['harm', "pain", "hurt", "damage", "violence", "loss", "vandalism"],
        'loyalty': ['loyalty', "faith", "support", "honesty", "honor", "devotion"],
        'non-moral': ['neutral'],
        'purity': ['purity', "clean", "pure", "clean", "pure"],
        'subversion': ['destruction', 'defeat', 'revolution']
    }
```