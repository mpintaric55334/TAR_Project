from sentence_transformers import SentenceTransformer
from data_manipulation.dataset import Features, PatientLabels, Notes, GenerationDataset
from model.model import LSTM, evaluate, train
import torch
from torch.utils.data import DataLoader, random_split
from model.loss import Loss

features = Features("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\features.csv")
labels = PatientLabels("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\train.csv", features)

encoder_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
notes = Notes("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\patient_notes.csv",
              labels.patient_dict, encoder_model)
dataset = GenerationDataset(notes.encoded, labels.patient_dict)
trainset, testset = random_split(dataset, [800, 200])
train_loader = DataLoader(trainset, batch_size=64, shuffle=True,
                          drop_last=True)
test_loader = DataLoader(testset, batch_size=1, shuffle=False,
                         drop_last=True)

print("Loaded data")


model = LSTM(num_layers=2, batch_size=64, dropout=0.2).to("cuda")
criterion = Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


model = train(model, train_loader, optimizer, criterion, 200)

model.reset_states()
evaluate(model, test_loader, "cuda")
