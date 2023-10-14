import datetime
import time
import math
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, total_steps, log_every=5000, logdir=None):
        self.logdir = logdir
        if self.logdir is None:
            self.logdir = "./logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.total_steps = total_steps
        self.log_every = log_every
        self.start = time.time()
        self.writer = SummaryWriter(self.logdir)
    
    def log(self, step, loss):
        self.writer.add_scalar("Train/Loss", loss, step)
        print("{step}/{total_steps} loss:{loss}".format(step=step, total_steps=self.total_steps, loss=loss))
    
    def logPredicted(self, step, log_text):
        self.writer.add_text("Eval/Prediction", log_text, step)

    def logAccuracy(self, step, accuracy):
        self.writer.add_scalar("Eval/Accuracy", accuracy, step)
    
    def logConfusionMatrix(self, step, fig):
        self.writer.add_figure("Eval/Confusion Matrix", fig, step)