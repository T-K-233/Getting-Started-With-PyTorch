import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def load(self, path):
        self.load_state_dict(torch.load(path))
    
    def save(self, path):
        torch.save(self.state_dict(), path)

import numpy as np
import json

class RNN_NP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def load(self, path):
        model_dict = json.load(open(path, "r"))
        self.layer_1_weights = np.array(model_dict["i2h.weight"])
        self.layer_1_biases = np.array(model_dict["i2h.bias"])
        self.layer_2_weights = np.array(model_dict["h2o.weight"])
        self.layer_2_biases = np.array(model_dict["h2o.bias"])
                
    def forward(self, input, hidden):
        layer_input = np.concatenate((input, hidden), axis=1)
        layer_1_output = np.matmul(layer_input, self.layer_1_weights.T) + self.layer_1_biases
        layer_2_output = np.matmul(layer_1_output, self.layer_2_weights.T) + self.layer_2_biases
        output = np.log(np.exp(layer_2_output) / np.sum(np.exp(layer_2_output), axis=1))
        print(output)
        quit()
        return output, layer_1_output
