import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import datetime

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from dataloader import Dataloader
from model import EncoderRNN
from logger import Logger
from trainer import EncoderTrainer


conf = OmegaConf.load("./configs/train.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = Dataloader(conf.dataloader.path)

print("Loading datasets:", Dataloader.findFiles('data/names/*.txt'))
dataloader.load()
print("")

# print(dataloader.unicodeToAscii('Ślusàrski'))
# print(dataloader.letterToTensor('J'))
# print(dataloader.lineToTensor('Jones').size())

# for i in range(10):
#     category, line, category_tensor, line_tensor = dataloader.getRandom()
#     print('category =', category, '/ line =', line)

model = EncoderRNN(dataloader.n_letters, conf.model.hidden_layer_size, dataloader.n_categories, device)


criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=conf.trainer.learning_rate)

logdir = "./logs/fit/" \
    + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") \
    + "_lr{lr}_hidden{hidden}".format(
        lr=conf.trainer.learning_rate, 
        hidden=conf.model.hidden_layer_size)

logger = Logger(
    total_steps=conf.trainer.n_iters,
    log_every=conf.logger.log_every,
    logdir=logdir
    )

trainer = EncoderTrainer(
    model=model, 
    criterion=criterion, 
    optimizer=optimizer,
    logger=logger,
    dataloader=dataloader)

trainer.train(conf.trainer.n_iters)


