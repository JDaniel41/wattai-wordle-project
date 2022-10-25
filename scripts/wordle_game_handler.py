import gym
import gym_wordle
from gym_wordle.envs.wordle_env import WORDS, encodeToStr, strToEncode
from typing import Tuple, List


class WordleGame():
    """
    A wrapper class for the game. We will use this to make guesses by integer and
    it will make it easier to integrate the gym environment with our model.
    """

    def __init__(self):
        self.env = gym.make('Wordle-v0')

    def reset_game(self):
        self.env.reset()
        self.num_turns_left = 6
        self.word_representation = [-1, -1, -1, -1, -1]
        self.guessed_words = set()
        print(f"Correct Word: {encodeToStr(self.env.hidden_word)}")

    def tokenize_word(self, guess, guess_result):
        """
        Update the tokenization with the most recent guess and return the tokenization.
        """
        old_tokenization = self.word_representation
        new_tokenization = []
        for old_symbol, new_symbol, result in zip(old_tokenization, guess, guess_result):
            if result == 2:
                new_tokenization.append(new_symbol)
            else:
                new_tokenization.append(old_symbol)
        self.word_representation = new_tokenization
        return new_tokenization

    def encode_alphabet(self, alphabet_state):
        """
        Change the encoding of the alphabet to look like

        -1 = Not Guessed Yet
        0 = Incorrect Guess. This is not in the word
        1 = It does appear in the word somewher.
        """
        encoded_alphabet = []
        for letter in alphabet_state:
            if letter == -1 or letter == 0:
                encoded_alphabet.append(letter)
            else:
                encoded_alphabet.append(1)
        return encoded_alphabet

    def get_current_state(self):
        """
        Return the current state of the game.
        """
        return self.encode_alphabet(self.env._get_obs()['alphabet']), self.word_representation, self.num_turns_left

    def already_guessed_word(self, word):
        return word in self.guessed_words

    def make_guess(self, word_num: int, debug_mode: bool = False) -> Tuple[List[int], int, int]:
        """
        Make a guess by word number. Updates the state internallly.

        :param word_num: The word number to guess.
        :param debug_mode: Whether to print debug information.

        :return: Boolean value of if we're done.
        """
        self.guessed_words.add(word_num)
        encoded_word = list(WORDS[word_num])
        print(f"Guess: {encodeToStr(encoded_word)}")
        if debug_mode:
            print(encodeToStr(encoded_word))
        obs, _, done, _ = self.env.step(encoded_word)

        self.tokenize_word(WORDS[word_num], obs['board']
                           [6-self.num_turns_left])
        self.num_turns_left -= 1
        return done