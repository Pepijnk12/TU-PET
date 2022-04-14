# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
To add a new task to PET, both a DataProcessor and a PVP for this task must
be added. The DataProcessor is responsible for loading training and test data.
This file shows an example of a DataProcessor for a new task.
"""

import json
import os
from typing import List, Dict, Optional

import torch

from TU_tweets.tweet_task_pvp import TweetTaskPVP
from pet.task_helpers import TaskHelper
from pet.tasks import DataProcessor, PROCESSORS, TASK_HELPERS, METRICS
from pet.utils import InputExample


class TweetDataProcessor(DataProcessor):
    """
    Example for a data processor.
    """

    # Set this to the name of the task
    TASK_NAME = "tweet-task"

    # Set this to the name of the file containing the train examples
    TRAIN_FILE_NAME = "data/train.jsonl"

    # # Set this to the name of the file containing the dev examples
    DEV_FILE_NAME = "data/val.jsonl"

    # Set this to the name of the file containing the test examples
    TEST_FILE_NAME = "test.jsonl"

    # Set this to the name of the file containing the unlabeled examples
    UNLABELED_FILE_NAME = "data/unlabeled.jsonl"

    # Set this to a list of all labels in the train + test data
    LABELS = ['authority', 'betrayal', 'care', 'cheating', 'degradation', 'fairness', 'harm', 'loyalty', 'non-moral', 'purity', 'subversion']


    def get_train_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads train examples from a file with name `TRAIN_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the training data can be found
        :return: a list of train examples
        """
        return self._create_examples(os.path.join(data_dir, TweetDataProcessor.TRAIN_FILE_NAME), "train")

    def get_dev_examples(self, data_dir: str) -> List[InputExample]:
        """
        This method loads dev examples from a file with name `DEV_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the dev data can be found
        :return: a list of dev examples
        """
        return self._create_examples(os.path.join(data_dir, TweetDataProcessor.DEV_FILE_NAME), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads test examples from a file with name `TEST_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the test data can be found
        :return: a list of test examples
        """
        return self._create_examples(os.path.join(data_dir, TweetDataProcessor.TEST_FILE_NAME), "test")

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """
        This method loads unlabeled examples from a file with name `UNLABELED_FILE_NAME` in the given directory.
        :param data_dir: the directory in which the unlabeled data can be found
        :return: a list of unlabeled examples
        """
        return self._create_examples(os.path.join(data_dir, TweetDataProcessor.UNLABELED_FILE_NAME), "unlabeled")

    def get_labels(self) -> List[str]:
        """This method returns all possible labels for the task."""
        return TweetDataProcessor.LABELS

    def _create_examples(self, path, set_type, max_examples=-1, skip_first=0):
        """Creates examples for the training and dev sets."""
        examples = []
        print(path)
        with open(path, encoding='utf8') as f:
            for idx, line in enumerate(f):
                json_line = json.loads(line)
                label = str(json_line['label']) if 'label' in json_line else None
                guid = "%s-%s" % (set_type, idx)
                tweet = json_line['tweet']
                example = InputExample(guid=guid, text_a=tweet, label=label, idx=idx)
                examples.append(example)

        return examples


# register the processor for this task with its name
PROCESSORS[TweetDataProcessor.TASK_NAME] = TweetDataProcessor
METRICS[TweetDataProcessor.TASK_NAME] = ["acc", "f1-macro"]

# Uncomment for multilabel metric
# METRICS[TweetDataProcessor.TASK_NAME] = ["multilabel"]


class TweetTaskHelper(TaskHelper):
    def train_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> Optional[torch.Tensor]:
        """
        Custom implementation of the train step for this task.

        :param batch: a batch of examples
        :return: a scalar loss tensor
        """
        inputs = self.wrapper.generate_default_inputs(batch)
        mlm_labels, labels = batch['mlm_labels'], batch['labels']

        # print(labels_set.size())

        outputs = self.wrapper.model(**inputs)
        prediction_scores = TweetTaskPVP(self.wrapper).convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])
        # print("predictions", prediction_scores)
        # print("mlm labels", mlm_labels)
        # print("normal labels", labels)
        # print(prediction_scores)
        # prediction_scores = self.wrapper.model(**inputs)[0].view(-1, self.wrapper.model.config.vocab_size)
        # prediction_scores = self.wrapper.model(**inputs)[0]
        bce = torch.nn.BCEWithLogitsLoss()
        # print("Prediction size", prediction_scores.size())
        # print(prediction_scores, mlm_labels)
        # print("Size predictions", prediction_scores.size())
        # print(prediction_scores, labels)
        bce_loss = bce(prediction_scores, labels.float())
        # print("BCE", bce_loss)
        return bce_loss


    def eval_step(self, batch: Dict[str, torch.Tensor], **kwargs) -> Optional[torch.Tensor]:
        """
        Custom implementation of the eval step for this task.

        :param batch: a batch of examples
        :return: a tensor of logits
        """
        pass

# optional: if you have to use verbalizers that correspond to multiple tokens, uncomment the following line
# TASK_HELPERS[TweetDataProcessor.TASK_NAME] = MultiMaskTaskHelper
TASK_HELPERS[TweetDataProcessor.TASK_NAME] = TweetTaskHelper
