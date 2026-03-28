import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

CATEGORY_NAMES = ["CUL", "FIN", "POL", "SCI", "SPO", "TEC"]


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} | Total: {total:,} | Percentage: {100 * trainable / total:.2f}%")
    return trainable, total


def evaluate(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=CATEGORY_NAMES)
    cm = confusion_matrix(all_labels, all_preds)

    print(report)
    print("Confusion matrix:")
    print(cm)

    return all_labels, all_preds, report, cm


def measure_latency(model, tokenizer, sample_text, device, max_length, iterations=50):
    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(**inputs)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iterations
    print(f"Average Inference Latency: {avg_ms:.2f} ms per document ({iterations} iterations)")
    return avg_ms
