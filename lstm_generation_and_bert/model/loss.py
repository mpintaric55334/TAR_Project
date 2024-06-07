import torch.nn as nn
import torch


class Loss:

    def __init__(self):

        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, predictions, true_sequence, masks):

        masks = masks.reshape(-1)

        predictions = predictions[:, :-1, :]
        predictions = predictions.reshape(-1, predictions.shape[2])

        true_sequence = true_sequence.long()
        true_sequence = true_sequence[:, 1:]
        true_sequence = true_sequence - 1
        true_sequence = true_sequence.reshape(-1)

        loss = self.criterion(predictions, true_sequence)

        loss = loss * masks

        return loss.sum() / masks.sum()