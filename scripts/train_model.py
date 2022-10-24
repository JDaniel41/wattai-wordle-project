from calendar import c
import torch
import gym
import gym_wordle
from gym_wordle.exceptions import InvalidWordException
import numpy as np
from gym_wordle.envs.wordle_env import WORDS, encodeToStr, strToEncode
from tqdm import tqdm
from torch.optim import Adam, SGD
from typing import Tuple, List
import wandb

import hyperopt
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials


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

# Define the Model Here


class WordleModel(torch.nn.Module):
    def __init__(self):
        super(WordleModel, self).__init__()

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
            torch.nn.Linear(57, 57*26),
            torch.nn.Dropout(),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(),
            # torch.nn.Linear(58*26, 58*26),
            # torch.nn.LeakyReLU(),
            # torch.nn.Dropout(),
            torch.nn.Linear(57*26, len(WORDS)),
        )

        # Output Softmax Layer
        self.output = torch.nn.Softmax(dim=0)

    def forward(self, x):
        # Input Layers
        x_valid_alphabet = self.valid_alphabet_letters(x[0])
        x_invalid_alphabet = self.invalid_alphabet_letters(x[1])
        x_current_word = self.current_word_state(x[2])
        x_num_turns_left = self.num_turns_left_state(x[3])

        # Hidden Layers
        # TODO: Add back the turns left layer
        x = torch.cat((x_valid_alphabet, x_invalid_alphabet,
                      x_current_word), dim=0)
        x = self.hidden_layer_stack(x)

        # Output Softmax Layer
        x = self.output(x)

        return x


def train_loop(model: WordleModel, epoch_num: int, loss_fn, optimizer):
    game = WordleGame()
    game.reset_game()
    # Play until the game ends.
    done = False

    labels = [0 if i == game.env.hidden_word else 1 for i in range(len(WORDS))]
    losses = []

    guessed_words = set()
    # Training Step
    while not done:
        # Get the current game state.
        alphabet, tokenization, turns_left = game.get_current_state()

        valid_alphabet_letters = []
        invalid_alphabet_letters = []

        for letter in alphabet:
            if letter == 1:
                valid_alphabet_letters.append(1)
                invalid_alphabet_letters.append(0)
            elif letter == 0:
                valid_alphabet_letters.append(0)
                invalid_alphabet_letters.append(1)
            else:
                valid_alphabet_letters.append(0)
                invalid_alphabet_letters.append(0)

        # Convert to tensors
        valid_alphabet_tensor = torch.tensor(
            valid_alphabet_letters, dtype=torch.float32)
        invalid_alphabet_tensor = torch.tensor(
            invalid_alphabet_letters, dtype=torch.float32)
        tokenization_tensor = torch.tensor(tokenization, dtype=torch.float)
        turns_left_tensor = torch.tensor([turns_left], dtype=torch.float)

        # Get the model's prediction for the current game state.
        output = model(
            [valid_alphabet_tensor, invalid_alphabet_tensor, tokenization_tensor, turns_left_tensor])

        # Get the word number with the highest probability that we haven't guessed already
        word_num = torch.argmax(output)
        weights = get_word_scores(output, game.env.hidden_word, guessed_words)
        weights = torch.tensor(weights, dtype=torch.float32)

        # Make the guess.
        done = game.make_guess(word_num)

        # Get the loss for the step
        loss = loss_fn(output,
                       weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append([epoch_num, loss.item()])
        wandb.log({"train_loss": loss.item(), "epoch": epoch_num})

        print(f"Guessed Word Score: {weights[word_num]}")
        print(f"Guessed Word Model Score: {output[word_num]}")

        guessed_words.add(word_num.item())

    return losses


def test_loop(model: WordleModel, epoch_num: int, loss_fn):
    game = WordleGame()
    game.reset_game()
    losses = []
    min_val_loss = 100000000000
    done = False
    guessed_words = set()
    while not done:
        alphabet, tokenization, turns_left = game.get_current_state()

        valid_alphabet_letters = []
        invalid_alphabet_letters = []

        for letter in alphabet:
            if letter == 1:
                valid_alphabet_letters.append(1)
                invalid_alphabet_letters.append(0)
            elif letter == 0:
                valid_alphabet_letters.append(0)
                invalid_alphabet_letters.append(1)
            else:
                valid_alphabet_letters.append(0)
                invalid_alphabet_letters.append(0)

        # Convert to tensors
        valid_alphabet_tensor = torch.tensor(
            valid_alphabet_letters, dtype=torch.float32)
        invalid_alphabet_tensor = torch.tensor(
            invalid_alphabet_letters, dtype=torch.float32)
        tokenization_tensor = torch.tensor(tokenization, dtype=torch.float)
        turns_left_tensor = torch.tensor([turns_left], dtype=torch.float)

        with torch.no_grad():
            # Get the model's prediction for the current game state.
            output = model(
                [valid_alphabet_tensor, invalid_alphabet_tensor, tokenization_tensor, turns_left_tensor])

            # Get the word number with the highest probability.
            word_num = torch.argmax(output).item()

            labels = get_word_scores(
                output, game.env.hidden_word, guessed_words)
            labels = torch.tensor(labels, dtype=torch.float)

            # Make the guess.
            done = game.make_guess(word_num)

            # Get the loss for the step
            loss = loss_fn(output, labels)

            losses.append([epoch_num, loss.item()])
            wandb.log({"test_loss": loss.item(), "epoch": epoch_num})

            if loss < min_val_loss:
                min_val_loss = loss

            guessed_words.add(word_num)
            print(f"Guessed Word Score: {labels[word_num]}")
            print(f"Guessed Word Model Score: {output[word_num]}")

    return losses, min_val_loss


def get_word_scores(model_out, correct_word, prev_guessed_words):
    # Compare the information helpfulness.
    # 0.2 points for a letter in the right spot, 0.1 points for a letter in the word but not in the right spot.
    # 0 points for a letter not in the word.
    # Each loss weight should be (1 - calculated_weight)
    scores = []

    print(prev_guessed_words)
    for idx, _ in enumerate(model_out):
        if idx in prev_guessed_words:
            scores.append(0)
            continue  # Don't guess a word we've already guessed.
        new_weight = 0
        encoding_of_guess = WORDS[idx]
        for guess_letter, actual_letter in zip(encoding_of_guess, correct_word):
            if guess_letter == actual_letter:
                new_weight += 0.2
            elif guess_letter in correct_word:
                new_weight += 0.1
        scores.append(new_weight)

    return scores


def train_model(model, epochs, learning_rate):
    train_losses = []
    test_losses = []

    wandb.config = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "early_stopping": 40,
    }

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(
    # ), lr=wandb.config['learning_rate'], weight_decay=wandb.config['weight_decay'])
    optimizer = SGD(model.parameters(), lr=wandb.config['learning_rate'])
    # Early Stopping
    best_loss = 100000000000
    early_stopping = 0
    best_model = None
    for i in range(epochs):
        # Reset the game for the new game.
        model.train()
        train_losses.append(train_loop(model, i, loss_fn, optimizer))

        # Test Eval Step
        model.eval()
        new_test_losses, min_test_loss = test_loop(model, i, loss_fn)

        # Early Stopping
        if min_test_loss < best_loss:
            best_loss = min_test_loss
            early_stopping = 0
            best_model = model.state_dict()
        else:
            early_stopping += 1
            if early_stopping > wandb.config['early_stopping']:
                print("No Improvement. Stoppoing Training.")
                break
        test_losses.append(new_test_losses)

    train_losses = np.array(train_losses).reshape(-1, 2)
    test_losses = np.array(test_losses).reshape(-1, 2)

    return min_test_loss, best_model


def objective(args):
    with wandb.init(project="wordle-watt-project", entity="jdaniel41", config=args):
        model = WordleModel()
        wandb.watch(model)
        min_test_loss, best_state_dict = train_model(
            model, args['epochs'], args['learning_rate'])
        print(min_test_loss)
        return {'loss': min_test_loss, 'status': STATUS_OK, 'attachments': {
            'model_state_dict': best_state_dict
        }}


if __name__ == '__main__':
    space = {
        'epochs': 1000,
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.1),
    }
    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=1,
                trials=trials)
    results = space_eval(space, best)
    print(results)
