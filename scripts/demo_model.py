from wordle_game_handler import WordleGame
from train_model import WordleModel
import torch

if __name__ == '__main__':
    model = WordleModel()
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    game = WordleGame()
    game.reset_game()

    done = False
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

        output = model(
            [valid_alphabet_tensor, invalid_alphabet_tensor, tokenization_tensor, turns_left_tensor])
        
        done = game.make_guess(torch.argmax(output))
        