import torch
import numpy as np
from torchvision.utils import make_grid

from base import BaseTrainer
from IPython.terminal.debugger import set_trace as keyboard

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader, valid_data_loader=None, lr_scheduler=None, 
                 train_logger=None, loss_weights=[1.0,0.1]):

        super(Trainer, self).__init__(model, loss, metrics, optimizer, 
                                      resume, config, train_logger)

        # initialize trainer parameters
        self.config = config
        self.data_loader = data_loader
        self.loss_weights = loss_weights
        self.lr_scheduler = lr_scheduler
        self.valid_data_loader = valid_data_loader
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.do_validation = self.valid_data_loader is not None

    def _eval_metrics(self, output, target):
        # evaluate model performance
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        # set model in training mode
        self.model.train()

        # initialize logging parameters
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        # loop over the dataset
        for batch_idx, (obs, act, epstarts) in enumerate(self.data_loader):
            # initialize hidden states
            self.model.hidden = self.model.init_hidden(epstarts=epstarts)

            # initialize optim
            self.optimizer.zero_grad()
            obs, act = obs.to(self.device), act.to(self.device)

            # perform prediction and compute loss
            output = self.model(obs)
            loss = self.loss(output, act, self.loss_weights)

            # perform backprop
            if self.config['arch']['mode'] == 'recurrent':
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            self.optimizer.step()

            # logging
            self.writer.set_step((epoch - 1)*len(self.data_loader) + batch_idx)
            self.writer.add_scalar('loss', loss.item())

            # append to loss
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, act)

        # log model performance
        log = {
            'loss': total_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        # perform validation
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        # perform lr update
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        # set model in eval mode
        self.model.eval()

        # initialize validation params
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        # validation loop
        with torch.no_grad():
            for batch_idx, (obs, act, epstarts) in enumerate(self.valid_data_loader):

                # initialize hidden states
                self.model.hidden = self.model.init_hidden(epstarts=epstarts)

                # initialize input
                obs, act = obs.to(self.device), act.to(self.device)

                # perform prediction and compute loss
                output = self.model(obs)
                loss = self.loss(output, act, self.loss_weights)

                # logging
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) \
                                    + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())

                # append to loss
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, act)

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
