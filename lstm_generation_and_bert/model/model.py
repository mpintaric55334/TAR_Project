import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from model.loss import Loss


class LSTM(nn.Module):
    def __init__(self, num_layers=2, batch_size=16, dropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=768, batch_first=True,
                            num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(768, 132)
        self.embedding = nn.Embedding(133, 20, padding_idx=0)
        self.states = (torch.zeros((num_layers, batch_size, 768), device="cuda"),
                       torch.zeros((num_layers, batch_size, 768), device="cuda"))
        self.num_layers = num_layers
        self.batch_size = batch_size

    def forward(self, inputs, encodings):
        encodings = encodings.unsqueeze(0)
        encodings = encodings.repeat(self.num_layers, 1, 1)
        self.states = (torch.zeros((self.num_layers, self.batch_size, 768),
                                   device="cuda"), encodings)
        embeddings = self.embedding(inputs.long())
        hiddens, _ = self.lstm(embeddings, self.states)
        outputs = self.fc(hiddens)
        return outputs

    def infer(self, inputs, encodings):

        if not self.infer_first_iter:
            encodings = encodings.repeat(self.num_layers, 1)
            self.states = (torch.zeros((self.num_layers,  768), device="cuda"),
                           encodings)
            self.infer_first_iter = True
        embeddings = self.embedding(inputs.long())
        x, self.states = self.lstm(embeddings, self.states)
        x = self.fc(x)
        return x
    
    def reset_states(self):
        self.states = (torch.zeros(self.num_layers, 768),  # or inference, batchs size 1
                       torch.zeros(self.num_layers, 768))
        self.infer_first_iter = False


def evaluate(model, testloader, device="cpu"):

    model.eval()
    predicted_all = []
    true_all = []

    for encodings, labels, masks in testloader:

        encodings = encodings.to("cuda")
        labels = labels.squeeze(0).to("cuda")
        masks = masks.to("cuda")
        next_tokens = torch.zeros((1), device=device)
        predicted_sequences = torch.zeros((labels.shape[0]), device=device)
        end_token_activated = False
        for idx in range(1, labels.shape[0]):
            outputs = model.infer(next_tokens, encodings)
            device = outputs.device
            next_tokens = outputs.argmax(dim=1) + 1
            if end_token_activated:
                next_tokens = torch.tensor([132])
            next_tokens = next_tokens.to(device)
            predicted_sequences[idx] = next_tokens[0]

            if next_tokens[0] == 132:
                end_token_activated = True

        model.reset_states()
        labels = labels[labels != 132]
        predicted_sequences = predicted_sequences[predicted_sequences != 132]
        labels = list(set(labels[1:].cpu().numpy()))
        predicted_sequences = list(set(predicted_sequences[1:].cpu().numpy()))
        length = 131

        binary_labels = [0] * length

        for index in labels:
            index = int(index)
            index -= 1
            if 0 <= index < length:
                binary_labels[index] = 1

        binary_predicted = [0] * length

        for index in predicted_sequences:
            index = int(index)
            index -= 1
            if 0 <= index < length:
                binary_predicted[index] = 1

        predicted_all.append(binary_predicted)
        true_all.append(binary_labels)

    macro_precision = precision_score(true_all, predicted_all, average='macro',
                                      zero_division=0)
    macro_recall = recall_score(true_all, predicted_all, average='macro',
                                zero_division=0)
    macro_f1 = f1_score(true_all, predicted_all, average='macro',
                        zero_division=0)

    micro_precision = precision_score(true_all, predicted_all, average='micro',
                                      zero_division=0)
    micro_recall = recall_score(true_all, predicted_all, average='micro',
                                zero_division=0)
    micro_f1 = f1_score(true_all, predicted_all, average='micro',
                        zero_division=0)

    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1


def train(dataset, params):

    model = LSTM(num_layers=params["num_layers"], batch_size=params["batch_size"],
                 dropout=params["dropout"]).to("cuda")

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=params["batch_size"],
                                               drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = Loss()

    for epoch in range(params["num_epochs"]):
        model.train()
        epoch_loss = 0
        n = 0
        for encodings, labels, masks in train_loader:

            encodings = encodings.to("cuda")
            labels = labels.to("cuda")
            masks = masks.to("cuda")

            outputs = model(labels, encodings)

            optimizer.zero_grad()

            loss = criterion.compute_loss(outputs, labels, masks)

            epoch_loss += loss.item()
            n += 1
            loss.backward()
            optimizer.step()

        print("Epoch", epoch, " loss is ", epoch_loss/n)
    return model


def train_with_validation(dataset, params):

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    average_metric = 0
    for _, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

        model = LSTM(num_layers=params["num_layers"], batch_size=params["batch_size"],
                     dropout=params["dropout"]).to("cuda")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=params["batch_size"],sampler=train_subsampler,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 sampler=val_subsampler)

        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion = Loss()

        for _ in range(params["num_epochs"]):
            model.train()
            epoch_loss = 0
            n = 0
            for encodings, labels, masks in train_loader:

                encodings = encodings.to("cuda")
                labels = labels.to("cuda")
                masks = masks.to("cuda")

                outputs = model(labels, encodings)

                optimizer.zero_grad()

                loss = criterion.compute_loss(outputs, labels, masks)

                epoch_loss += loss.item()
                n += 1
                loss.backward()
                optimizer.step()

        model.reset_states()
        _, _, macro_f1, _, _, micro_f1 = evaluate(model, val_loader, "cuda")
        average_metric = macro_f1 * micro_f1

    return average_metric
