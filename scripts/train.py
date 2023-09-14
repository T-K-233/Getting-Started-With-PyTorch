import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from dataloader import Dataloader
from model import RNN
from logger import Logger
from trainer import Trainer


conf = OmegaConf.load('./configs/train.yaml')

dataloader = Dataloader(conf.dataloader.path)

print(dataloader.findFiles('data/names/*.txt'))
print(dataloader.unicodeToAscii('Ślusàrski'))
print(dataloader.letterToTensor('J'))
print(dataloader.lineToTensor('Jones').size())

for i in range(10):
    category, line, category_tensor, line_tensor = dataloader.randomTrainingExample()
    print('category =', category, '/ line =', line)


model = RNN(dataloader.n_letters, conf.model.hidden_layer_size, dataloader.n_categories)


criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=conf.trainer.learning_rate)

logger = Logger()

trainer = Trainer(
    model=model, 
    criterion=criterion, 
    optimizer=optimizer,
    logger=logger,
    dataloader=dataloader)


trainer.train(conf.trainer.n_iters)

model.save(conf.model.path)



