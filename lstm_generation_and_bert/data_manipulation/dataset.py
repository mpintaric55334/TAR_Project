import pandas as pd
from torch.utils.data import Dataset
import torch


class Features:

    def __init__(self, feature_csv):

        df = pd.read_csv(feature_csv)
        df = df.drop(["case_num"], axis=1)
        self.original_idx_dict = {}
        self.idx_to_feature = {}
        self.feature_to_idx = {}
        idx = 1

        for _, row in df.iterrows():
            self.original_idx_dict[row[0]] = row[1]

            if row[1] not in self.feature_to_idx:
                self.feature_to_idx[row[1]] = idx
                idx += 1

        for key in self.feature_to_idx:
            self.idx_to_feature[self.feature_to_idx[key]] = key


class PatientLabels:

    def __init__(self, notes_csv, features):

        df = pd.read_csv(notes_csv)
        df = df.drop(["id", "case_num"], axis=1)

        df = df[(df['annotation'] != "[]") & (df['location'] != "[]")]
        df = df.drop(["annotation", "location"], axis=1)

        self.patient_dict = {}
        self.features = features
        self.bert_labels = {}

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

        for pn_num in self.patient_dict:

            label_list = [0] * 131
            
            for index in self.patient_dict[pn_num]:
                index = int(index)
                index -= 1
                if 0 <= index <= 131:
                    label_list[index] = 1
                    
            self.bert_labels[pn_num] = label_list


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
        self.masks = []
        MAX_NUM_FEATURES = 18

        for key in encoding_dict:
            self.encodings.append(encoding_dict[key])
            feature_labels = label_dict[key]
            feature_labels.insert(0, 0)

            mask = []
            mask.extend([1] * len(feature_labels))
            if len(feature_labels) < 18:
                extension_num = (MAX_NUM_FEATURES + 1 - len(feature_labels))
                feature_labels.extend([132] * extension_num)
                mask.append(1)
                mask.extend([0] * (extension_num - 1))
            
            mask = mask[1:]  #  to be decided
            self.labels.append(feature_labels)
            self.masks.append(mask)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        return torch.Tensor(self.encodings[index]), torch.Tensor(self.labels[index]), torch.Tensor(self.masks[index])