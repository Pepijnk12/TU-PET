### First iteration of model:
- Accuracy validation set: 50% (not fair because 90% prediction is non-moral)
- 32 random training samples (most of them non-moral)
- 100 random validation samples
- 6775 unlabeled samples
- The model predicts non-moral 90% of the time

<hr>

###Patterns:
    p1 = text_a, '. This made me feel:', self.mask], []
    p2 = ['My tweet is: ', text_a, '. Therefore I believe in', self.mask], []
    p3 = ['My tweet is: ', text_a, '. Therefore ', self.mask, 'is important to me'], []

- acc-p1: 0.5467 +- 0.0252
- acc-p2: 0.5367 +- 0.0153
- acc-p3: 0.49 +- 0.0265
- acc-all-p: 0.5244 +- 0.0328
