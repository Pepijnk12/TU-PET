## Thrid iteration of model:

### Impportant:
- Correct pre-processing
- Removed duplicate tweets
- Same data distribution train and validation data

### Summary
- Accuracy validation set: 80,9%
- 100 train samples of 50 or more characters
- 1000 validation samples of 50 or more characters
- 10000 unlabeled samples
- ```roberta-base``` model

**Train data distribution:**
```{"authority": 3, "betrayal": 0, "care": 11, "cheating": 8, "degradation": 1, "fairness": 8, "harm": 16, "loyalty": 7, "non-moral": 43, "purity": 0, "subversion": 3, "total_moral": 57, "total": 100}```

**Validation data distribtion:** ```{"authority": 32, "betrayal": 5, "care": 113, "cheating": 81, "degradation": 13, "fairness": 79, "harm": 166, "loyalty": 75, "non-moral": 436, "purity": 0, "subversion": 0, "total_moral": 564, "total": 1000}```

```{"train_set_before_training": 0.03, "global_step": 7500, "average_loss": 0.2642059664947291, "train_set_after_training": 0.85, "test_set_after_training": {"acc": 0.809}}```

###Patterns:
    p1 = [text_a, . This made me feel:, self.mask], []
    p2 = [My tweet is: , text_a, . Therefore I believe in, self.mask], []
    p3 = [My tweet is: , text_a, . Therefore , self.mask, is important to me], []

    acc-p1: 0.751 +- 0.011357816691600558
    acc-p2: 0.7656666666666667 +- 0.004163331998932269
    acc-p3: 0.7563333333333333 +- 0.015011106998930282
    acc-all-p: 0.7576666666666667 +- 0.01158663022625648

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
