import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset


class Features:

    def __init__(self, feature_csv):

        df = pd.read_csv(feature_csv)
        df = df.drop(["case_num"], axis=1)
        self.original_idx_dict = {}
        self.idx_to_feature = {}
        self.feature_to_idx = {}
        idx = 0

        for _, row in df.iterrows():
            self.original_idx_dict[row[0]] = row[1]

            if row[1] not in self.feature_to_idx:
                self.feature_to_idx[row[1]] = idx
                idx += 1

        for key in self.feature_to_idx:
            self.idx_to_feature[self.feature_to_idx[key]] = key

        print(len(self.feature_to_idx))


def is_empty_list(lst):
    return len(lst) == 0


class PatientLabels:

    def __init__(self, notes_csv, features):

        df = pd.read_csv(notes_csv)
        df = df.drop(["id", "case_num"], axis=1)

        df = df[(df['annotation'] != "[]") & (df['location'] != "[]")]
        df = df.drop(["annotation", "location"], axis=1)

        self.patient_dict = {}
        self.features = features

        for _, row in df.iterrows():

            pn_num = row["pn_num"]
            feature_num = row["feature_num"]
            feature_text = self.features.original_idx_dict[feature_num] 
            new_feature_num = self.features.feature_to_idx[feature_text]
            if pn_num not in self.patient_dict:
                self.patient_dict[pn_num] = [new_feature_num]
            else:
                feature_list = self.patient_dict[pn_num]
                feature_list.append(new_feature_num)
                self.patient_dict[pn_num] = feature_list


class Notes:

    def __init__(self, notes_csv, patient_dict, encoder_model):

        df = pd.read_csv(notes_csv)
        df = df.drop("case_num", axis=1)
        self.texts = {}
        self.encoded = {}
        for _, row in df.iterrows():
            if row[0] in patient_dict:
                text = row[1].replace("\n", " ").replace("\r", "").lower()
                self.texts[row[0]] = text
                self.encoded[row[0]] = encoder_model.encode(text)

     
class GenerationDataset(Dataset):

    def __init__(self, encoding_dict, label_dict):
        self.encodings = []
        self.labels = []
        self.mask = []

        for key in encoding_dict:
            self.encodings.append(encoding_dict[key])
            self.labels.append(label_dict[key])

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        return self.encodings[index], self.labels[index]


features = Features("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\features.csv")
labels = PatientLabels("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\train.csv", features)

encoder_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
notes = Notes("C:\\Users\\Matija\\Desktop\\APT_PROJEKT\\TAR_Project\\data\\patient_notes.csv",
              labels.patient_dict, encoder_model)

dataset = GenerationDataset(notes.encoded, labels.patient_dict)
