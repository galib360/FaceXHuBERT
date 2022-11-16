import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
from tqdm import tqdm
from transformers import Wav2Vec2Processor
import librosa    


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data, subjects_dict, data_type="train"):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.emo_one_hot_labels = np.eye(2)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        sentence_id = int(file_name.split(".")[0].split("_")[1])
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        if sentence_id > 40:
            emo_one_hot = self.emo_one_hot_labels[1]
        else:
            emo_one_hot = self.emo_one_hot_labels[0]

        return torch.FloatTensor(audio), vertice, torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name, torch.FloatTensor(emo_one_hot)

    def __len__(self):
        return self.len


def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    vertices_path = os.path.join(args.dataset, args.vertices_path)
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft") # HuBERT uses the processor of Wav2Vec 2.0

    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_path = os.path.join(r,f)
                speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                input_values = processor(speech_array, return_tensors="pt", padding="longest", sampling_rate=sampling_rate).input_values
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                    # print("Vertices Data Not Found! ", vertice_path)
                else:
                    data[key]["vertice"] = vertice_path

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    splits = {'BIWI': {'train': list(range(1, 37)) + list(range(41, 77)), 'val': list(range(37, 39)) + list(range(77, 79)), 'test': list(range(39, 41)) + list(range(79, 81))}}
   
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict


def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)
    train_data = Dataset(train_data, subjects_dict,"train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(valid_data, subjects_dict,"val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(test_data, subjects_dict,"test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    return dataset


if __name__ == "__main__":
    get_dataloaders()

    