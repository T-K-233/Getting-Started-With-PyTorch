
from io import open
import glob
import os
import unicodedata
import string

import numpy as np
import torch
import random

class Dataloader:
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        for filename in self.findFiles(self.dataset_path):
            category = os.path.splitext(os.path.basename(filename))[0]
            self.all_categories.append(category)
            lines = self.readLines(filename)
            self.category_lines[category] = lines

        self.n_categories = len(self.all_categories)


    def findFiles(self, path):
        return glob.glob(path)

    def categoryFromOutput(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i


    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    # Read a file and split into lines
    def readLines(self, filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]


    # Find letter index from all_letters, e.g. "a" = 0
    def letterToIndex(self, letter):
        return self.all_letters.find(letter)

    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def letterToTensor(self, letter):
        tensor = torch.zeros(1, self.n_letters)
        tensor[0][self.letterToIndex(letter)] = 1
        return tensor

    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li, letter in enumerate(line):
            tensor[li][0][self.letterToIndex(letter)] = 1
        return tensor
    
    # Turn a line into a <line_length x 1 x n_letters>,
    # or an array of one-hot letter vectors
    def lineToNumpy(self, line):
        array = np.zeros((len(line), 1, self.n_letters))
        for li, letter in enumerate(line):
            array[li][0][self.letterToIndex(letter)] = 1
        return array


    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingExample(self):
        category = self.randomChoice(self.all_categories)
        line = self.randomChoice(self.category_lines[category])
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self.lineToTensor(line)
        return category, line, category_tensor, line_tensor



    # Get a random category and random line from that category
    def randomTrainingPair(self):
        category = self.randomChoice(self.all_categories)
        line = self.randomChoice(self.category_lines[category])
        return category, line

    # One-hot vector for category
    def categoryTensor(self, category):
        li = self.all_categories.index(category)
        tensor = torch.zeros(1, self.n_categories)
        tensor[0][li] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(self, line):
        tensor = torch.zeros(len(line), 1, self.n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][self.all_letters.find(letter)] = 1
        return tensor

    # ``LongTensor`` of second letter to end (EOS) for target
    def targetTensor(self, line):
        letter_indexes = [self.all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(self.n_letters - 1) # EOS
        return torch.LongTensor(letter_indexes)
    
    # Make category, input, and target tensors from a random category, line pair
    def randomTrainingExample2(self):
        category, line = self.randomTrainingPair()
        category_tensor = self.categoryTensor(category)
        input_line_tensor = self.inputTensor(line)
        target_line_tensor = self.targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor