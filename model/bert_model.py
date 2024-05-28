from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from transformers import Trainer
from torch.nn import BCEWithLogitsLoss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.tensor(logits)).numpy() > 0.5
    macro_precision = precision_score(labels, predictions, average='macro',
                                      zero_division=0)
    macro_recall = recall_score(labels, predictions, average='macro',
                                zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)

    micro_precision = precision_score(labels, predictions, average='micro',
                                      zero_division=0)
    micro_recall = recall_score(labels, predictions, average='micro',
                                zero_division=0)
    micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
    return {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
    }


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss