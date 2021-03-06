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
be added. The PVP is responsible for applying patterns to inputs and mapping
labels to their verbalizations (see the paper for more details on PVPs).
This file shows an example of a PVP for a new task.
"""
import json
from typing import List

from pet.pvp import PVP, PVPS
from pet.utils import InputExample


class TweetTaskPVP(PVP):
    """
    Example for a pattern-verbalizer pair (PVP).
    """

    # Set this to the name of the task
    TASK_NAME = "tweet-task"


    USE_BIG_VERBALIZER = True
    if USE_BIG_VERBALIZER:
        with open("TU_tweets/verbalizer_data/big_verbalizer_single_token.json", "r") as f:
            VERBALIZER = json.load(f)
            for k, v in VERBALIZER.items():
                print(k, len(v))
            VERBALIZER['non-moral'] = ['neutral']
    else:
        VERBALIZER = {
            'authority': ['authority', 'force', 'government', 'jurisdiction', 'rule'],
            'betrayal': ['betrayal', 'deception', 'dishonest', 'treason'],
            'care': ['care', 'responsibility', 'protection', 'trust'],
            'cheating': ['cheating', 'lying', 'unfair'],
            'degradation': ['degradation', 'deception', 'fraud', 'dishonest', 'dishonest'],
            'fairness': ['fairness', "equality", "equal", "fair", "justice", "honesty", "integrity", "balanced",
                         "truth"],
            'harm': ['harm', "pain", "hurt", "damage", "violence", "loss", "vandalism"],
            'loyalty': ['loyalty', "faith", "support", "honesty", "honor", "devotion"],
            'non-moral': ['neutral'],
            'purity': ['purity', "clean", "pure", "clean", "pure"],
            'subversion': ['destruction', 'defeat', 'revolution']
        }

    def get_parts(self, example: InputExample):
        """
        This function defines the actual patterns: It takes as input an example and outputs the result of applying a
        pattern to it. To allow for multiple patterns, a pattern_id can be passed to the PVP's constructor. This
        method must implement the application of all patterns.
        """

        # We tell the tokenizer that both text_a and text_b can be truncated if the resulting sequence is longer than
        # our language model's max sequence length.
        text_a = self.shortenable(example.text_a)

        patterns = {
            1: ([text_a, '. This made me feel:', self.mask], []),
            2: (['My tweet is: ', text_a, '. Therefore I believe in', self.mask], []),
            3: (['My tweet is: ', text_a, '. Therefore ', self.mask, 'is important to me'], []),
            4: (["I think that:", text_a, '. This made me feel ', self.mask], []),
            5: ([text_a, '. This makes me feel', self.mask], []),
            6: ([text_a, '. I think that ', self.mask], []),
            7: ([text_a, '. I feel ', self.mask, ' about it'], []),
            8: ([text_a, '. I am ', self.mask], [])
        }

        # For each pattern_id, we define the corresponding pattern and return a pair of text a and text b (where text b
        # can also be empty).
        if self.pattern_id in patterns:
            return patterns[self.pattern_id]
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return TweetTaskPVP.VERBALIZER[label]


# register the PVP for this task with its name
PVPS[TweetTaskPVP.TASK_NAME] = TweetTaskPVP
