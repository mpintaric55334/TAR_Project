from data_manipulation.bert_dataset import BertNotes, BertDataset
from data_manipulation.dataset import Features, PatientLabels
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from model.bert_model import compute_metrics, CustomTrainer
from torch.utils.data import random_split
import torch

torch.manual_seed(42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=131)


features = Features("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\features.csv")
labels = PatientLabels("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\train.csv", features)

notes = BertNotes("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\patient_notes.csv",
                  labels.patient_dict)

dataset = BertDataset(notes.texts, labels.bert_labels, tokenizer)
trainset, testset = random_split(dataset, [800, 200])

training_args = TrainingArguments(
    output_dir="output_dir",
    num_train_epochs=30,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=trainset,
    eval_dataset=testset,
    compute_metrics=compute_metrics
)


trainer.train()
results = trainer.evaluate()
print(results)
