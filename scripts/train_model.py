from calendar import c
import torch
from gym_wordle.exceptions import InvalidWordException
import numpy as np
from gym_wordle.envs.wordle_env import WORDS, encodeToStr, strToEncode
from tqdm import tqdm
from torch.optim import Adam, SGD
from typing import Tuple, List
import wandb
import pickle
import os

import hyperopt
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials

from wordle_game_handler import WordleGame




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


def train_loop(model: WordleModel, epoch_num: int, loss_fn, optimizer, device):
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
            valid_alphabet_letters, dtype=torch.float32).to(device)
        invalid_alphabet_tensor = torch.tensor(
            invalid_alphabet_letters, dtype=torch.float32).to(device)
        tokenization_tensor = torch.tensor(tokenization, dtype=torch.float).to(device)
        turns_left_tensor = torch.tensor([turns_left], dtype=torch.float).to(device)

        # Get the model's prediction for the current game state.
        output = model(
            [valid_alphabet_tensor, invalid_alphabet_tensor, tokenization_tensor, turns_left_tensor]).to(device)

        # Get the word number with the highest probability that we haven't guessed already
        word_num = torch.argmax(output).to(device)
        weights = get_word_scores(output, game.env.hidden_word, guessed_words)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)

        # Make the guess.
        done = game.make_guess(word_num)

        # Get the loss for the step
        loss = loss_fn(output,
                       weights).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append([epoch_num, loss.item()])
        wandb.log({"train_loss": loss.item(), "epoch": epoch_num})

        print(f"Guessed Word Score: {weights[word_num]}")
        print(f"Guessed Word Model Score: {output[word_num]}")

        guessed_words.add(word_num.item())

    return losses


def test_loop(model: WordleModel, epoch_num: int, loss_fn, device):
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
            valid_alphabet_letters, dtype=torch.float32).to(device)
        invalid_alphabet_tensor = torch.tensor(
            invalid_alphabet_letters, dtype=torch.float32).to(device)
        tokenization_tensor = torch.tensor(tokenization, dtype=torch.float).to(device)
        turns_left_tensor = torch.tensor([turns_left], dtype=torch.float).to(device)

        with torch.no_grad():
            # Get the model's prediction for the current game state.
            output = model(
                [valid_alphabet_tensor, invalid_alphabet_tensor, tokenization_tensor, turns_left_tensor]).to(device)

            # Get the word number with the highest probability.
            word_num = torch.argmax(output).to(device).item()

            labels = get_word_scores(
                output, game.env.hidden_word, guessed_words)
            labels = torch.tensor(labels, dtype=torch.float).to(device)

            # Make the guess.
            done = game.make_guess(word_num)

            # Get the loss for the step
            loss = loss_fn(output, labels).to(device)

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    model.to(device)
    for i in range(epochs):
        # Reset the game for the new game.
        model.train()
        train_losses.append(train_loop(model, i, loss_fn, optimizer, device))

        # Test Eval Step
        model.eval()
        new_test_losses, min_test_loss = test_loop(model, i, loss_fn, device)

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

    # train_losses = np.array(train_losses).reshape(-1, 2)
    # test_losses = np.array(test_losses).reshape(-1, 2)

    wandb.log({"min_test_loss": min_test_loss})

    return min_test_loss, best_model


def objective(args):
    with wandb.init(project="wordle-watt-project", entity="jdaniel41", config=args):
        model = WordleModel()
        wandb.watch(model, log_freq=10, log='all')
        min_test_loss, best_state_dict = train_model(
            model, args['epochs'], args['learning_rate'])
        print(min_test_loss)
        torch.cuda.empty_cache()
        return {'loss': min_test_loss, 'status': STATUS_OK, 'model_state_dict': best_state_dict}


if __name__ == '__main__':
    space = {
        'epochs': 1000,
        'learning_rate': hp.uniform('learning_rate', 0.0001, 0.1),
    }
    trials = Trials()

    if os.path.exists('wordle_trials.trials'):
        trials = pickle.load(open("wordle_trials.trials", "rb"))

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=200,
                trials=trials)
    results = space_eval(space, best)

    valid_trial_list = [trial for trial in trials.results if STATUS_OK == trial['status']]
    losses = [float(trial['loss']) for trial in valid_trial_list]

    min_loss_index = np.argmin(losses)

    best_model = valid_trial_list[min_loss_index]['model_state_dict']
    torch.save(best_model, 'best_model.pth')

    pickle.dump(trials, open('wordle_trials.trials', "wb"))
