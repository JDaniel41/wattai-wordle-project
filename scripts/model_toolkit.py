import torch


class WordleModel(torch.nn.Module):
    def __init__(self, num_words: int, hidden_multiplier: int = 26):
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
            torch.nn.Linear(57, 57*hidden_multiplier),
            torch.nn.Dropout(),
            torch.nn.Linear(57*hidden_multiplier, num_words),
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
