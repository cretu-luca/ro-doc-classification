import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from MOROCO.loadDataSet import loadMOROCODataSet


class MOROCODataset(Dataset):
    def __init__(self, samples, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            samples,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([l - 1 for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_data(config):
    data_prefix = config["dataset"]["moroco"]["data_prefix"]
    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]

    data = loadMOROCODataSet(data_prefix)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    splits = {}
    for split_name in ("train", "validation", "test"):
        ids, samples, dialect_labels, category_labels = data[split_name]
        splits[split_name] = {
            "dataset": MOROCODataset(samples, category_labels, tokenizer, max_length),
            "samples": samples,
            "category_labels": category_labels,
        }

    return splits, tokenizer
