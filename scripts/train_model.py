import torch
import numpy as np
from gym_wordle.envs.wordle_env import WORDS, encodeToStr, strToEncode
from torch.optim import Adam, SGD
from typing import Tuple, List
import wandb
import pickle
import os

from wordle_game_handler import WordleGame
from model_toolkit import WordleModel


def get_state_tensors(game, device):
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
    tokenization_tensor = torch.tensor(
        tokenization, dtype=torch.float).to(device)
    turns_left_tensor = torch.tensor(
        [turns_left], dtype=torch.float).to(device)

    return valid_alphabet_tensor, invalid_alphabet_tensor, tokenization_tensor, turns_left_tensor


def get_guess(model, game, device):
    # Get the current game state.
    valid_alphabet_tensor, invalid_alphabet_tensor, tokenization_tensor, turns_left_tensor = get_state_tensors(
        game, device)

    # Get the model's prediction for the current game state.
    output = model(
        [valid_alphabet_tensor, invalid_alphabet_tensor, tokenization_tensor, turns_left_tensor]).to(device)

    # Get the word number with the highest probability.
    word_num = torch.argmax(output).to(device).item()

    return word_num, output


def get_word_labels(correct_word, prev_guessed_words, current_tokenized, current_alphabet):
    scores = []

    for idx, _ in enumerate(WORDS):
        if idx in prev_guessed_words:
            scores.append(0)
            continue  # Don't guess a word we've already guessed.
        new_weight = 0
        encoding_of_guess = WORDS[idx]
        for letter_idx, (guess_letter, actual_letter) in enumerate(zip(encoding_of_guess, correct_word)):
            if guess_letter == actual_letter:
                # We got a NEW LETTER in the right spot.
                if current_tokenized[letter_idx] == -1:
                    new_weight += 0.2
            elif guess_letter in correct_word:
                new_weight += 0.1
            elif current_alphabet[guess_letter] == -1:
                # We guessed a new letter that's not in the word.
                new_weight += 0.1
        scores.append(new_weight)

    return scores


def play_game_with_model(model, game: WordleGame, device, guessed_words, loss_fn):
    guess, model_out = get_guess(model, game, device)

    labels = get_word_labels(
        game.env.hidden_word, guessed_words, game.word_representation, game.get_current_alphabet())

    labels = torch.tensor(labels, dtype=torch.float32).to(device)

    # Make the guess.
    done, is_win = game.make_guess(guess)

    # Get the loss for the step
    loss = loss_fn(model_out,
                   labels).to(device)

    print(f"Guessed Word Score: {labels[guess]}")
    print(f"Guessed Word Model Score: {model_out[guess]}")

    guessed_words.add(guess)

    return loss, done, guess, model_out, is_win


def train_loop(model: WordleModel, epoch_num: int, loss_fn, optimizer, device):
    game = WordleGame()
    total_loss = 0
    model.train()
    num_wins = 0

    for i in range(wandb.config['train_games_per_epoch']):
        print(f"Train Game {i}")
        guessed_words = set()
        done = False
        game.reset_game()
        guess_num = 0
        while not done:
            loss, done, _, _, is_win = play_game_with_model(
                model, game, device, guessed_words, loss_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item(), "epoch": epoch_num,
                      "guess_num": guess_num, "game_num": i})

            total_loss += loss.item()

            if is_win:
                print("WON A GAME!")
                num_wins += 1
        wandb.log({"num_guessed_words_train": len(guessed_words)})

    wandb.log({
        "avg_train_loss": total_loss / wandb.config['train_games_per_epoch'],
        "epoch": epoch_num
    })
    wandb.log({"num_train_wins": num_wins})

    return total_loss / wandb.config['train_games_per_epoch'], num_wins


def test_loop(model: WordleModel, epoch_num: int, loss_fn, device):
    game = WordleGame()
    game.reset_game()
    total_test_loss = 0
    num_wins = 0
    model.eval()

    # Let's play some games! We will report the average test loss to WANDB.
    for i in range(wandb.config['test_games_per_epoch']):
        print(f"Test Game {i}")
        guessed_words = set()
        done = False
        guess_num = 0
        while not done:
            with torch.no_grad():
                # Get the model's prediction for the current game state.
                loss, done, _, _, is_win = play_game_with_model(
                    model, game, device, guessed_words, loss_fn)

                wandb.log({
                    "guess_test_loss": loss.item(),
                    "epoch": epoch_num,
                    "guess_num": guess_num,
                    "game_num": i
                })

                if is_win:
                    print("WON A GAME!")
                    num_wins += 1

                total_test_loss += loss.item()
            guess_num += 1

        game.reset_game()
        wandb.log({"num_guessed_words_test": len(guessed_words)})

    wandb.log({
        "avg_test_loss": total_test_loss / wandb.config['test_games_per_epoch'],
        "epoch": epoch_num
    })

    wandb.log({
        'num_test_wins': num_wins,
        'epoch': epoch_num
    })

    return total_test_loss / wandb.config['test_games_per_epoch'], num_wins


def train_model():
    run = wandb.init(project="wordle-watt-project", entity="jdaniel41", )
    model = WordleModel(
        len(WORDS), wandb.config['hidden_multiplier'])

    print(f"CONFIG: {wandb.config}")

    wandb.watch(model, log_freq=1000, log='all')

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = Adam(model.parameters(
    # ), lr=wandb.config['learning_rate'], weight_decay=wandb.config['weight_decay'])
    optimizer = SGD(model.parameters(), lr=wandb.config['learning_rate'])
    # Early Stopping
    best_loss = 100000000000
    early_stopping = 0
    best_model = None
    total_test_wins = 0
    total_train_wins = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    model.to(device)
    for i in range(wandb.config['epochs']):
        # Reset the game for the new game.
        model.train()
        avg_train_loss, num_train_wins = train_loop(
            model, i, loss_fn, optimizer, device)

        # Test Eval Step
        model.eval()
        avg_test_loss, num_test_wins = test_loop(model, i, loss_fn, device)

        total_train_wins += num_train_wins
        total_test_wins += num_test_wins

        wandb.log({
            "total_train_wins": total_train_wins,
            "total_test_wins": total_test_wins,
            "total_wins": total_train_wins + total_test_wins,
            "epoch": i
        })

        # Early Stopping
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            early_stopping = 0
            best_model = model.state_dict()
        else:
            early_stopping += 1
            if early_stopping > wandb.config['early_stopping']:
                print("No Improvement. Stoppoing Training.")
                break

    wandb.run.summary["min_test_loss"] = best_loss
    torch.save(best_model.state_dict(), 'best_model.pth')
    artifact = wandb.Artifact(name='best_model.pth', type='model')
    wandb.run.log_artifact(artifact)

    torch.cuda.empty_cache()
    return


if __name__ == '__main__':
    wandb.agent('jdaniel41/wordle-watt-project/zf6jdgps',
                function=train_model, count=100)
