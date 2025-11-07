import torch
import numpy as np
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        
    def __call__(self, val_loss, model, epoch, optimizer, rank):

        score = -val_loss

        # early stopping called for the first time
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        # the score is lower than the previous one
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if (self.patience > -1) and (self.counter >= self.patience) :
                self.early_stop = True
        # the score is higher than the previous one
        else:
            self.best_score = score
            self.counter = 0
            if rank == 0:
                self.save_checkpoint(val_loss, model, epoch, optimizer)
            

    def save_checkpoint(self, val_loss, model, epoch, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, f"model_save-{epoch}.pt"))
        torch.save(optimizer.state_dict(), os.path.join(self.path, "model_save-optim-{}.pt".format(epoch)))
        print(f'Model checkpoint saved at epoch {epoch}')
        self.val_loss_min = val_loss