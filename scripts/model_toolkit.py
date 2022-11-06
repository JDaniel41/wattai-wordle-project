import torch
from gym_wordle.envs.wordle_env import WORDS
from typing import Tuple, List, Union


class InvalidWordProcessorLayer(torch.nn.Module):
    def __init__(self, words: List[Tuple[int, int, int, int, int]]):
        super(InvalidWordProcessorLayer, self).__init__()
        self.words = words
        self.num_words = len(words)
        self.unadjusted_letters = set(range(0, 26))

        # Make a set of 26 arrays. Each of them has 0 or 1 multipliers for
        # each word. Basically, if we know that 'a' cannot be in the word, we take
        # the tensor for 'a' and multiply that by the WORDS array. This should
        # strip out all words that don't have that letter. Lots of pre-processing
        # time potentially, but this makes it run fast in forward() since we're
        # just doing tensor multiplication.

        # These tensors will have a 1 if the letter doesn't appear in the word.
        # and a 0 if the letter does appear in the word.
        self.letter_filters = torch.ones(26, self.num_words)

        for word_num, word in enumerate(words):
            for letter in word:
                self.letter_filters[letter][word_num] = 0

    def to(self, device):
        self.letter_filters = self.letter_filters.to(device)
        return self

    def forward(self, probs: torch.Tensor, invalid_letters: torch.Tensor) -> torch.Tensor:
        # At this point, the input should be the softmax probabilities?
        # If the word contains any invalid letter, we need to remove it.
        # We can do this by setting the probability of the word to 0.
        output = probs.clone()
        for idx, is_invalid in enumerate(invalid_letters):
            if is_invalid:
                # Multiply the output tensor by that letter filter
                output *= self.letter_filters[idx]
        return output


class WordleModel(torch.nn.Module):
    def __init__(self, words: List[Tuple[int, int, int, int, int]], hidden_multiplier: int = 26):
        super(WordleModel, self).__init__()
        self.num_words = len(words)
        self.words = words

        # Input Layers
        self.valid_alphabet_letters = torch.nn.Linear(
            26, 26)   # 0 for unknown, 1 in word
        self.invalid_alphabet_letters = torch.nn.Linear(
            26, 26)  # 0 for unknown, 1 for not in word
        self.current_word_state = torch.nn.Linear(5, 5)
        self.num_turns_left_state = torch.nn.Linear(1, 1)

        # Hidden Layers
        # TODO: Change to 58 if we add back turns left.
        self.hidden_layer_stack = torch.nn.Sequential(
            torch.nn.Linear(57, 57*hidden_multiplier),
            torch.nn.Dropout(),
            torch.nn.Linear(57*hidden_multiplier, self.num_words),
        )

        # Output Softmax Layer
        self.softmax_output = torch.nn.Softmax(dim=0)

        self.invalid_word_processor = InvalidWordProcessorLayer(words)

    def reset_word_processor(self):
        self.invalid_word_processor.reset()

    def to(self, device):
        super(WordleModel, self).to(device)
        self.invalid_word_processor.to(device)

    def forward(self, x):
        # Input Layers
        y_valid_alphabet = self.valid_alphabet_letters(x[0])
        y_invalid_alphabet = self.invalid_alphabet_letters(x[1])
        y_current_word = self.current_word_state(x[2])
        y_num_turns_left = self.num_turns_left_state(x[3])

        # Hidden Layers
        # TODO: Add back the turns left layer
        y = torch.cat((y_valid_alphabet, y_invalid_alphabet,
                      y_current_word), dim=0)
        y = self.hidden_layer_stack(y)

        # Output Softmax Layer
        y = self.softmax_output(y)
        y = self.invalid_word_processor(y, x[1])

        return y
