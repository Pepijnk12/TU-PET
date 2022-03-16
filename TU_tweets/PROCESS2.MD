## Second iteration of model:
### Summary
- Accuracy validation set: ??
- 5 samples per label (55 in total) random training samples (most of them non-moral)
- 200 random validation samples
- 6775 unlabeled samples
- ```roberta-base``` model

#### Compared to process 1:
- Only difference with first iteration is the training set.
- Training set contains 5 samples per label (5 samples * 11 labels = 55 tweets) instead of 32 random tweets

```train_set_before_training": 0.09090909090909091, "global_step": 5082, "average_loss": 0.1317346813170029, "train_set_after_training": 0.5272727272727272, "test_set_after_training": {"acc": 0.285}}```

###Patterns:
    p1 = text_a, . This made me feel:, self.mask], []
    p2 = [My tweet is: , text_a, . Therefore I believe in, self.mask], []
    p3 = [My tweet is: , text_a, . Therefore , self.mask, is important to me], []

- acc-p1: 0.2567 +- 0.0486
- acc-p2: 0.2683 +- 0.0388
- acc-p3: 0.2717 +- 0.0252
- acc-all-p: 0.2656 +- 0.0342

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
