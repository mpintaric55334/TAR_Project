from sentence_transformers import SentenceTransformer
from data_manipulation.dataset import Features, PatientLabels, Notes, GenerationDataset
from model.model import LSTM, evaluate, train, train_with_validation
from torch.utils.data import DataLoader, random_split
import torch
torch.manual_seed(0)

features = Features("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\features.csv")
labels = PatientLabels("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\train.csv", features)

encoder_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
notes = Notes("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\patient_notes.csv",
              labels.patient_dict, encoder_model)
dataset = GenerationDataset(notes.encoded, labels.patient_dict)
trainset, testset = random_split(dataset, [800, 200])
test_loader = DataLoader(testset, batch_size=1, shuffle=False,
                         drop_last=True)

print("Loaded data")


layers = [1, 2]
num_epochs = [50, 100, 120]
lrs = [1e-2, 1e-3, 1e-4]
batch_sizes = [32, 64]
dropouts = [0, 0.2]
i = 0
"""
best_params = {}
best_average_metric = 0
for layer in layers:
    for epochs in num_epochs:
        for lr in lrs:
            for batch_size in batch_sizes:
                for dropout in dropouts:
                    params = {"num_layers": layer, "num_epochs": epochs,
                              "lr": lr, "batch_size": batch_size,
                              "dropout": dropout}
                    average_metric = train_with_validation(dataset, params)
                    if average_metric > best_average_metric:
                        best_average_metric = average_metric
                        best_params = params
                    print(i)
                    i += 1

print(best_params, best_average_metric)
"""
best_params = {"num_layers": 2, "num_epochs": 95, "lr": 1e-2,
               "batch_size": 32, "dropout": 0.2}
model = train(trainset, best_params)

model.reset_states()
macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1 = evaluate(model, test_loader, "cuda")


print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")

print(f"Micro Precision: {micro_precision:.4f}")
print(f"Micro Recall: {micro_recall:.4f}")
print(f"Micro F1 Score: {micro_f1:.4f}")
