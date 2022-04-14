---
layout: post
title:  "Reproduction of “It’s Not Just Size That Matters” by using the Moral Foundations Twitter Corpus"
date:   2022-04-14 22:12:08 +0200
categories: reproduction
---

# Reproduction of "It's Not Just Size That Matters"
###### Group 6 - Authors:
###### Ivor Zagorac, Pepijn Klop, Jason Qiu, Luca Cras 

### Introduction
In this blog post, we discuss the reproduction of the paper *It’s Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners* by T. Schick and H. Schütze[^pet-2]. The paper is an update of a paper by the same authors, where they introduce a novel model for few-shot learning, which has orders of magnitude fewer trainable parameters than GPT-3. They call this new method *pattern-exploiting-training* (PET). As can be seen in figure 1, the researchers showed that, by using PET, it was possible to achieve few-shot text classification performance similar to GPT-3 on SuperGLUE with LMs that had *three orders of magnitude fewer parameters*. The authors also developed an iterative version of PET (iPET)[^pet], but in this blog post, we will limit our discussion and reproduction to the base PET implementation.

![](https://i.imgur.com/sjoeNwQ.png =500x)
*Figure 1: iPET vs. PET vs. GPT-3, SuperGLUE performance*

In short, PET is a semi-supervised training procedure that reformulates input examples as cloze-style phrases to help language models understand a given task. These phrases are then used to assign soft labels to a large set of unlabeled examples, thereby increasing the overall training set size. Finally, standard supervised training is performed on the resulting training set. This definition is rather vague, so let us explain this further by an example.

Taking the example given in the original paper[^pet], let us look at the task of identifying whether two sentences $a$ and $b$ contradict each other (label $y_0$) or agree with each other ($y_1$). 

If we consider $M$ to be a masked language model, with vocabulary $V$ and a mask token: $\_\_\_\_ ∈ V$. PET uses a *pattern*, which is a function $P$ that takes as input a sequence of phrases $x = (s1,...,sk)$ with $s_i ∈ V^*$ as input, in order to output a phrase or sentence $P(x) ∈ V^∗$ that contains one or more mask tokens, i.e. a cloze question.

A cloze question is one where a passage of text is given, with one or multiple words removed/masked. In our example, we would use a pattern $P(a, b) = a? \_\_\_\_$, $b.$

To solve this cloze question, PET defines a *verbalizer* as an injective function $v : L → V$ that maps each label $y ∈ L$ to a word from $M$’s vocabulary. Going back to our example we construct a verbalizer $v$ that maps $y_0$ to “Yes” and $y_1$ to “No”. 

Given an example input pair:
> x = (Mia likes pie, Mia hates pie)

PET no longer has to assign a label without any inherent meaning, and can instead find the label $y ∈ L$ for which $v(y)$ is the most likely substitute for the mask. In the question:
> $P(x) =$ Mia likes pie? ____, Mia hates pie.

PET can now answer the more meaningful question of whether "Yes" or "No" is more likely to appear at the masked position, instead of having to answer whether a certain label ($y_0$ or $y_1$) fits better. Another example, this time for entailment classification, can be seen below in figure 1.

![](https://i.imgur.com/MLMB8nN.png =500x)
*Figure 2: Pattern-Verbalization-Pair, a schematic overview*


Now that you have an intuitive understanding of the basic concepts that PET takes advantage of, let us discuss the next steps. In this blog post, our main aim is to improve upon PET by providing an option for multi-label training and classification. In other words, we want to find out if, using PET, a language model can correctly assign multiple labels where necessary. To achieve this, we use a tweet dataset that has been annotated with one or more moral values. We use different methods of distilling these annotations into labels and we discuss the results from our multi-label classification experiments and check whether these results are still comparable to the results of the original paper. Finally, we finish off with a discussion of this reproduction, the reproducibility of the paper[^pet-2], and a conclusion.

### Dataset

Because our needs for this reproduction differed from the original paper, i.e. we needed data with multiple labels, we used a different dataset. Specifically, it was the *Moral Foundations Twitter Corpus (MTFC)*[^mftc], which is a collection of 35,108 tweets. These tweets are drawn from seven different domains and annotated with 10 moral sentiment labels plus 1 non-moral label. Initially, the data records are stored in a .json file named [MFTC_V4.json](https://osf.io/cwu4m/). Each tweet data in this .json file contains a tweet ID and different label annotations from three to four trained annotators, and each annotator may give one or more sentiment labels to this tweet. The annotated labels can be roughly categorized into five contrastive pairs.
- Pair 1: Care/Harm
- Pair 2: Fairness/Cheating
- Pair 3: Loyalty/Betrayal
- Pair 4: Authority/Subversion
- Pair 5: Purity/Degradation
- Non-moral

This dataset was suitable for our task because it was naturally multi-labelled at the first glance, but it could also be converted to a single-label dataset (say, by majority voting). This dataset allowed us to reproduce our paper in the single-label and multi-label manners without changing a different dataset. Also, tweet data are varied and strongly personalized with the user's sentiment, which made the annotation of MFTC a lot easier with high confidence. Considering that our PET model did not need a very large amount of data, the MFTC was already an ideal choice.

However, the dataset was separated from the original tweet texts, so we fetched them by a pre-written script to get the data we needed. Also, tweet texts are messy in nature and we responded to this issue with a wise tweet processing tool, which turned out to improve our accuracy to a great extent. Along with the processing, a noteworthy thing is that we filtered the tweet data and only kept those with a length of at least 60 characters. This helped reduce the potential drawbacks brought by super short tweets. Another issue was that it was not realistic to use all labels from all annotators. We specified a way of choosing among all the available annotations in both single-label and multi-label settings which will be explained later in the blog.

![](https://i.imgur.com/P4y7unJ.png =600x)
*Figure 3: A tweet and its annotation labels in MFTC_V4.json*

Following the above procedure, we stored our processed tweet data in .jsonl files. Each line of a .jsonl file contains a json object encapsulating a tweet text and an entry of label(s). Some examples are shown below.

```
{"tweet": "yep serve and protect is now obey and respect scandal scandal abc all lives matter", "label": "authority"}
{"tweet": "at user that is right gov show these idiots what government is supposed to do for the people bipartisanship", "label": "authority"}
{"tweet": "at user you re a walking talking example of social media done right great model for other govt officials of using social to lead", "label": "authority"}
...
```

With these processed tweet data, we split the data into a training set of 100 tweets, a validation of 1000 tweets and an unlabeled dataset of 10000 tweets. This is because we adjusted the number of tweets within the training set and determined that 100 tweets was a good balance between not using too many samples and having a decent accuracy. Also, in the multi-label setting, we add more labels to the entry of the json object. In this way, the tweets could finally be taken as inputs to the semi-supervised model.

### Methodology

As was already explained, the original PET framework has been created to do single-label tasks. So before we could implement a multi-label classification method, we first wanted to test whether the framework was able to do single-label classification on our newly presented dataset. To do this we had to implement the several parts of the PET framework as specified earlier. The explanation of how to implement PET for your own task is explained in the authors Github repo[^pet-github]

However, the dataset consists of annotations instead of labels, therefore we first needed to specify a way of labeling the tweets using the annotations provided in the dataset. In the single-label setting, we decided to use two different labeling techniques. The simplest technique is to filter the tweets such that only tweets that have been unanimously labeled by the annotators are left. This filtering left us with enough data to test the PET framework so we used it to run our first experiments which will be shown later in the blog. Another technique we implemented is majority vote. With this technique we would label the tweets with the annotation that has been chosen most by the annotators. If all annotators chose different annotations for a tweet then we would filter out this particular tweet. Again this left us with enough data to use the PET framework and experiments with this set will be shown later in the blog post.

In the multi-label setting, we also had some different ways to label our data. One was an optimistic way by keeping all labels with two or more votes. Once the number of votes achieved 2, we believed such labels were valid. The other was an pessimistic way by keeping only the label(s) with the maximum count. In this way, we could only see multiple labels once there was a tie in the label counts and the tie gave a count of at least two. In either optimistic or pessimistic way, we could see some tweets with different labels but all the labels only had counts of one. This implies that the annotators were not consistent with each other, and we decided to put such tweets into the unlabeled set. In our latest setting, we adopted the optimistic way of assigning multiple labels.

There was still another design choice we had to make namely on which language model to use. Since PET is a general framework that is not specified for a specific language model, we had the freedom of choosing between language models such as albert[^albert], bert[^bert], roberta[^roberta] and others. We chose roberta as our language model because it is shown that it performs really well. It would be interesting to do further research into using other language models but that is out of the scope of this reproduction. Now that we labeled the data and decided on a language model, it was time to adapt the PET framework, so create a DataProcessor and a PVP, to work with the MFTC dataset[^mftc].

#### DataProcessor
The implementation of the DataProcessor is pretty straight forward. Its job is solely to load training and test data. Becuase this blog post is about explaining what we did and not about showing code, it is unnecessary to show the exact implementation of the DataProcessor. It is important to mention that the training and test sets were created based on the design choices already specified in the Dataset section.

#### PVP
As was mentioned earlier, the PVP consists of two main components: the verbalizer and the patterns. 

##### Verbalizer
We created the first version of a verbalizer manually by finding synonyms for the labels. This approach did lead to some troubles, however.

Language models use tokenizers to split words into tokens which are then in turn used by the model as inputs. The roberta model use a BPE tokenizer[^bpe].These tokens can be single words but more often they are parts of words, e.g. terrible is split into terri- and -ble. The PET framework requires that the words in the verbalizer are single-token words because each mask is a single token. Thus we had to filter the verbalizer but even after the filtering we had multiple synonyms left per label which later showed to perform well on the dataset.

Another solution to the single-token problem is to use the MultiMask feature that is already implemented in the PET framework. Although this seems like an easy way to fix the problem, this feature led to a significant increase in run-time. Because this feature is still in development, the maximum batch size for evaluation is equal to one which means we are not able to run it in parallel. Additionally, the time required to run the evaluation scales about linearly with the amount of tokens for the longest verbalizer. These downsides to the MultiMask feature and the fact that the filtered verbalizer was performing well suggested that there was no reason for us to use the feature. 

Additionally, we used the Moral Foundations Dictionary 2.0 (MFD 2.0)[^mfd] as a verbalizer. This is a dictionary that consists of words and phrases that have a similar meaning to the moral values that we use as labels for our tweets. The advantage of this dictionary is that it is created by evaluating language models and testing whether they consider the words and phrases to be similar to the moral values. This means that in our classification tasks the model is likely to predict these dictionary entries for the labels. The disadvantage on the other hand is that MFD 2.0 consists of a lot of entries. The amount of synonyms per label is ~200. This could lead to the model having too many possible predictions and therefore not being able to choose the right one. In the results we will discuss whether this verbalizer had a positive or a negative effect.

##### Patterns
In the paper by Schick and Schutze, the authors specify multiple language tasks with patterns for each task. The table below shows some examples of patterns for these tasks.

|<!-- -->|<!-- -->
|---------| -------
| Pattern for determining to which word pronoun $p$ refers in sentence $s$: ![](https://i.imgur.com/chp2sKh.png) | Pattern for deciding whether a word $w$ has the same meaning in sentence $s_1$ and $s_2$:![](https://i.imgur.com/tKShX0W.png)
| Pattern for answering a yes/no question $q$ based on a passage $p$:![](https://i.imgur.com/sDvrGwm.png) | Pattern for deciding whether an answer $a$ is the correct answer to a question $q$ based on a passage $p$: ![](https://i.imgur.com/lsw48Xv.png)

For our task we created patterns that have the same kind of layout as the ones presented in the research paper. Additionally, we used our own common knowledge about language and expressing moral values to come up with patterns that allow the model to easily predict moral values. The exact patterns are presented in the Results section.

Now that we had a full implementation of PET for the single-label classification of moral values in tweets, we were able to start implementing the multi-label classification.

#### Multi-Label
<!-- Explain how we first did the classification with the logits
Explain that we tried to do multi-label but that it was not possible with the current code of PET -->
When implementing single-label classification, we found out that PET is really targeted towards single-label classification. Therefore our first approach was a naive way of testing whether it was possible to use the single-label model to do multi-label classification. From the model we could get a confidence score of each label for each tweet. So for each tweet we now knew how likely a certain label is according the model that was trained with single-labels. We then normalized these outputs with a sigmoid function. Now that we have a normalized confidence score, we can apply a threshold for multi-label classification. This means that when a confidence score is higher than the threshold, the corresponding label will be given to the tweet. Thus a low threshold will lead to every tweet having a high amount of labels and a high threshold will lead to fewer labels per tweet. The results of having different thresholds for the classification are shown in the results.

The next step was training the data with multiple labels and testing whether PET would still perform well. The authors of the paper have some documentation on their GitHub of how to create a TaskHelper which could help us to implement our multi-label classification. There was, however, nothing specified about these TaskHelpers in the paper and the documentation in the code was also lacking which made it very difficult to actually implement them. After digging deep into the code and finding what we had to modify such as creating a binary cross-entropy loss function it looked like it might be possible to do the multi-label classification. But unfortunately the code was setup in such a way that with the time we had for this reproduction we were unable to fully implement a multi-label classification. This still could be interesting for further research.


### Experiments and Results


We ran multiple experiments with different configurations for both single non-binary label classification as well as multi label classification.

We started off by running single-label experiments using different patterns, but we found that as long as the patterns tried to convey similar messages (i.e. not "I feel ____" vs. "He plays ____") resulted in similar accuracies.

Pattern 1: \<tweet>. This made me feel: \<mask>.
Pattern 2: My tweet is: \<tweet>. Therefore I believe in, \<mask>.
Pattern 3: My tweet is: \<tweet>. Therefore, \<mask> is important to me.
Pattern 4: I think that: \<tweet>. This made me feel: \<mask>.
Pattern 5: \<tweet>. This makes me feel \<mask>.
Pattern 6: \<tweet>. I think that \<mask>.
Pattern 7: \<tweet>. I feel \<mask> about it.
Pattern 8: \<tweet>. I am \<mask>.

|           | Accuracy | F1 macro |
|-----------|----------|----------|
| Pattern 1 | 0.72     | 0.50     |
| Pattern 2 | 0.73     | 0.52     |
| Pattern 3 | 0.72     | 0.48     |
| Pattern 4 | 0.74     | 0.49     |
| Pattern 5 | 0.71     | 0.48     |
| Pattern 6 | 0.68     | 0.38     |
| Pattern 7 | 0.71     | 0.44     |
| Pattern 8 | 0.69     | 0.42     |

In the further experiments we only used the first three patterns.

#### Verbalizers

We also experimented with two different verbalizers, as discussed in the methodology section. We hypothesized that using a verbalizer that was larger and, therefore, more elaborative would be beneficial to the accuracy. However, we found that the large verbalizer offered (at most) only a negligible difference.


*Single label classification accuracy with two different verbalizers:*
|                  | Accuracy |
|------------------|----------|
| Synonym Verbalizer | 0.80     |
| MFD 2.0 Verbalizer   | 0.80     |


Finally, we conducted some experiments for multi-label classification. In these experiments, the resulting accuracy and F1 scores dropped significantly. This result, however, can, in our opinion, largely be ascribed to the fact that we have been unable to implement multi-label training for the PET procedure. The original code that was provided by the authors turned out to be very difficult to adapt for our use case.



#### Datasets
*Single label classification accuracy with two different datasets:*
|                                                 | Accuracy |
|-------------------------------------------------|----------|
| Dataset with real world distribution            | 0.80     |
| Dataset with at least 5 samples for every label | 0.78     |


#### Single and multi label classification
*Single and multi label classifcation metrics:*
|              | Exact accuracy | Weighted F1 score | Macro F1 score |
|--------------|----------------|-------------------|----------------|
| Single label | 0.80           | 0.80             | 0.52           |
| Multi label  | 0.35           | 0.46              | 0.30           |

*Multi label classifcation F1 score for different thresholds:*
![](https://i.imgur.com/pRiBycH.jpg)
The best threshold to use is 0.93, which has an. f1 score of 0.46.


### Conclusion and Discussion
To conclude, the results of single label classification were in line with those of the original paper, even with different verbalizers, patterns, and data. 

Furthermore, we found that, even without training on multiple labels, we can still predict multiple labels sometimes, albeit with a much lower accuracy. An accuracy of 0.35 is still about four times higher than completely random which would amount to 1 / 11 (0.091%).

Finally, we found that the original results in the paper were easy to reproduce. The authors publicly shared the code and documented the project well. However, when trying to adapt the project for multi-label training, we quickly found out that the project was definitely not made with this in mind, as the code that needed changing was completely undocumented, hard to understand, and rigid.

With more time, adapting PET for multi-label classification and comparing the results with larger SOTA models such as GPT-3 would definitely be an interesting topic for future research / reproductions. 

### Work division
Over the course of our reproducibility project, we worked as a group by planning weekly meetings and discussing our changes with the TA. When the excel of papers came online, we quickly decided that we would be interested in reproducing a paper in the NLP field. The dataset was already chosen for us, so that was convenient. Everybody read the paper and took a look at the code so we all understood the inner workings. We then started on creating a planning for the project, that we have tried to stick to but midway through, we found out that  we wanted a slight course correction so we deviated a little from the initial desktop. Eventually, Pepijn ran most of the experiments because his laptop was capable of running the training in a decent amount of time. We were mostly present, but we compensated a little by having the other three write a little more on this blog post and proofreading. Overall, we are very happy with how the project went, although we unfortunately did not have enough time for the poster session, in part, due to exams.

<!-- 
#### What is the value of doing reproduction?

#### Do our reproduction results uphold the conclusions of the paper -->

### References

[^pet]: Timo Schick and Hinrich Schütze. "Exploiting Cloze Questions for Few-Shot Text Classification and Natural Language Inference". In: CoRR abs/2001.07676 (2020). arXiv: 2001.07676. URL: https://arxiv.org/abs/2001.07676 
[^pet-2]: Timo Schick and Hinrich Schütze. "It's Not Just Size That Matters: Small Language Models Are Also Few-Shot Learners". In: CoRR abs/2009.07118 (2020). arXiv: 2009.07118. URL: https://arxiv.org/abs/2009.07118 
[^albert]: Zhenzhong Lan et al. "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations". In: CoRR abs/1909.11942 (2019). arXiv: 1909.11942. URL: https://arxiv.org/abs/1909.11942
[^bert]: Jacob Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". In: CoRR abs/1810.04805 (2018). arXiv: 1810.04805. URL: https://arxiv.org/abs/1810.04805
[^roberta]: Yinhan Liu et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach". In: CoRR abs/1907.11692 (2019). arXiv:1907.11692. URL: https://arxiv.org/abs/1907.11692
[^bpe]: Rico Sennrich, Barry Haddow, and Alexandra Birch. "Neural Machine Translation of Rare Words with Sub-word Units". In: CoRR abs/1508.07909 (2015). arXiv: 1508.07909. URL: https://arxiv.org/abs/1508.07909
[^mftc]: Morteza Dehghani et al. Moral Foundations Twitter Corpus. URL: https://osf.io/k5n7y/
[^mfd]: Jeremy Frimer. Moral Foundations Dictionary 2.0. URL: https://osf.io/ezn37/
[^pet-github]: Timo Schick et al. Pattern-Exploiting Training (PET). URL: https://github.com/timoschick/pet