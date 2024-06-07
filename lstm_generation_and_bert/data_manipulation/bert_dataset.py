import torch
import pandas as pd


class BertNotes:

    def __init__(self, notes_csv, patient_dict):

        df = pd.read_csv(notes_csv)
        df = df.drop("case_num", axis=1)
        self.texts = {}
        self.encoded = {}
        for _, row in df.iterrows():
            if row[0] in patient_dict:
                text = row[1].replace("\n", " ").replace("\r", "").lower()
                self.texts[row[0]] = text


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, text_dict, labels_dict, tokenizer):
        texts = []
        self.labels = []
        for pn_num in text_dict:
            texts.append(text_dict[pn_num])
            self.labels.append(labels_dict[pn_num])
        self.labels = torch.tensor(self.labels)
        self.encodings = tokenizer(texts, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt")

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)
